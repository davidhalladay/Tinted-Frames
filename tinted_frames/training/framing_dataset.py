"""
Framing Consistency Dataset

Loads LLaVA-1.5 mix 665k training data and expands each sample into a quadruplet:
  - open-ended (original, no postfix)
  - open_constrained (original question + short-answer postfix, GPT short answer)
  - yes/no (rephrased + random postfix)
  - MCQ (4-choice, rephrased + random postfix)

Yes/no, MCQ, and short_answer variants are generated on-the-fly and cached to disk in
question_mapper.json (keyed by sample id).

Two LLM backends are supported:
  - "gpt"   : OpenAI API (default gpt-4o-mini); uses response_format=json_object
  - "vllm"  : local vLLM server (OpenAI-compatible); uses --vllm_host/port/model.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# Postfix pools
# ---------------------------------------------------------------------------

YESNO_POSTFIXES = [
    " Please answer yes or no.",
    " Answer with yes or no only.",
    " Respond with either yes or no.",
    " Give a yes or no answer.",
    "\nPlease answer with yes or no.",
]

MCQ_POSTFIXES = [
    "\nPlease return only the option letter.",
    "\nAnswer with the option letter only.",
    "\nChoose the correct option and return only its letter.",
    "\nSelect the correct answer (letter only).",
    "\nRespond with only the letter of the correct answer.",
]

OPEN_CONSTRAINED_POSTFIXES = [
    "\nPlease try to answer the question with short words or phrases if possible.",
    "\nPlease answer the question using a single word or phrase.",
    "\nGive a short answer.",
]


# ---------------------------------------------------------------------------
# Prompt for framing generation
# ---------------------------------------------------------------------------

FRAMING_SYSTEM = """You are a helpful assistant that reframes open-ended visual questions into yes/no and MCQ formats. Always respond with valid JSON only."""


def _build_framing_prompt(question: str, answer: str, should_affirm: bool) -> str:
    if should_affirm:
        yes_no_instruction = f"Convert to a binary yes/no question where the answer is 'yes' (affirming that the answer is '{answer}')."
        expected_yn = "yes"
    else:
        yes_no_instruction = f"Convert to a binary yes/no question where the answer is 'no' by replacing the correct answer with a plausible but incorrect alternative (negating that the answer is '{answer}')."
        expected_yn = "no"

    return f"""Given the following question and its correct answer, follow these steps to create framings:

Original Question: {question}
Correct Answer: {answer}

STEP 1 - Short Answer:
   - First, create a concise short answer (1-5 words) for the original question
   - This should be a brief, direct phrase — no full sentences

STEP 2 - Open-Ended Question:
   - Rephrase the original question as a natural open-ended question
   - Use WH-questions (What/Which/Where/Who/How/Why/etc.)

STEP 3 - Yes/No Question:
   - {yes_no_instruction}
   - Use the short answer to construct the yes/no question
   - Must be a binary question (Is/Are/Does/Do/Can/Could/etc.)

STEP 4 - MCQ (Multiple Choice Question):
   - Convert to a WH-question that asks for identification
   - Provide exactly 4 options where ONE option MUST contain the short answer from Step 1
   - The other 3 options should be plausible distractors

IMPORTANT RULES:
- Generate short answer FIRST, then use it for open-ended, yes/no and MCQ
- One MCQ option MUST contain/match the short answer
- For MCQ distractors: prioritize diversity and distinctiveness

Output ONLY valid JSON in this exact format:
{{
  "short_answer": "<concise 1-5 word phrase>",
  "open_ended": {{
    "question": "What/Which/Where/Who/How/Why..."
  }},
  "yes_no": {{
    "question": "Is/Are/Does/Do... question based on short_answer",
    "answer": "{expected_yn}"
  }},
  "mcq": {{
    "question": "What/Which/Where... question",
    "options": ["option1", "option2", "option3", "option4 (one must contain short_answer)"]
  }}
}}"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(client, model: str, system: str, user: str, backend: str, max_retries: int = 3) -> Optional[str]:
    """Call GPT or vLLM with retries. Returns raw response text or None."""
    if client is None:
        return None
    extra_kwargs = {"response_format": {"type": "json_object"}} if backend == "gpt" else {}

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra_kwargs,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[FramingDataset] LLM call ({backend}) failed after {max_retries} attempts: {e}")
                return None


def _parse_json(text: str, model_name: Optional[str] = None) -> Optional[Dict]:
    """Parse JSON, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _find_correct_option_index(short_answer: str, options: List[str]) -> int:
    """Return index (0-3) of the option that contains the short answer (case-insensitive)."""
    short_ans = short_answer.strip().lower()
    for i, opt in enumerate(options):
        opt_lower = opt.strip().lower()
        if short_ans in opt_lower or opt_lower in short_ans:
            return i
    return 0


def _generate_framings(client, model: str, question: str, answer: str, backend: str) -> Dict[str, Any]:
    """Single combined API call -> returns short_answer, open-ended, yes/no, and MCQ variants."""
    should_affirm = random.random() < 0.5
    expected_yn = "yes" if should_affirm else "no"

    prompt = _build_framing_prompt(question, answer, should_affirm)
    response = _call_llm(client, model, FRAMING_SYSTEM, prompt, backend)

    if response is not None:
        data = _parse_json(response, model_name=model)
        if data and "short_answer" in data and "open_ended" in data and "yes_no" in data and "mcq" in data:
            short_answer = data["short_answer"]
            open_ended = data["open_ended"]
            yn = data["yes_no"]
            mcq = data["mcq"]

            if "question" in open_ended and "question" in yn and "question" in mcq and "options" in mcq:
                yn["answer"] = expected_yn
                options = mcq["options"]
                correct_index = _find_correct_option_index(short_answer, options)
                answer_letter = chr(65 + correct_index)

                return {
                    "open_ended": {"question": open_ended["question"]},
                    "yesno": {"question": yn["question"], "answer": expected_yn},
                    "mcq": {
                        "question": mcq["question"],
                        "options": options,
                        "answer_index": correct_index,
                        "answer_letter": answer_letter,
                    },
                    "short_answer": short_answer,
                }

    return _fallback_framings(question, answer, expected_yn)


def _fallback_framings(question: str, answer: str, expected_yn: str) -> Dict[str, Any]:
    """Fallback when LLM generation fails."""
    short_answer = answer[:50] if len(answer) > 50 else answer
    if expected_yn == "yes":
        yn_q = f"Is it true that: {question.lower()}"
    else:
        yn_q = f"Is it false that: {question.lower()}"

    return {
        "open_ended": {"question": question},
        "yesno": {"question": yn_q, "answer": expected_yn},
        "mcq": {
            "question": question,
            "options": [short_answer, "unknown option 1", "unknown option 2", "unknown option 3"],
            "answer_index": 0,
            "answer_letter": "A",
        },
        "short_answer": short_answer,
    }


# ---------------------------------------------------------------------------
# Cache manager (multi-process safe)
# ---------------------------------------------------------------------------

class QuestionMapperCache:
    """Multi-process safe JSON cache for generated framing variants using file locking."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.lock_path = cache_path + ".lock"
        self._cache: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if FileLock is None:
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "r") as f:
                        self._cache = json.load(f)
                except (json.JSONDecodeError, OSError):
                    self._cache = {}
            return

        lock = FileLock(self.lock_path, timeout=30)
        for attempt in range(3):
            try:
                with lock:
                    if os.path.exists(self.cache_path):
                        with open(self.cache_path, "r") as f:
                            self._cache = json.load(f)
                return
            except (json.JSONDecodeError, OSError) as e:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                    self._cache = {}
                else:
                    self._cache = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]):
        self._cache[key] = value

        if FileLock is None:
            tmp_path = self.cache_path + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(self._cache, f, indent=2)
                os.replace(tmp_path, self.cache_path)
            except OSError:
                pass
            return

        lock = FileLock(self.lock_path, timeout=30)
        tmp_path = self.cache_path + f".tmp.{os.getpid()}"

        for attempt in range(3):
            try:
                with lock:
                    if os.path.exists(self.cache_path):
                        try:
                            with open(self.cache_path, "r") as f:
                                disk_cache = json.load(f)
                            disk_cache.update(self._cache)
                            self._cache = disk_cache
                        except (json.JSONDecodeError, OSError):
                            pass
                    with open(tmp_path, "w") as f:
                        json.dump(self._cache, f, indent=2)
                    os.replace(tmp_path, self.cache_path)
                return
            except OSError:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def __len__(self):
        return len(self._cache)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FramingDataset(Dataset):
    """
    LLaVA-665k dataset that expands each sample into a (open, open_constrained, yesno, mcq) quadruplet.
    """

    def __init__(
        self,
        data_path: str,
        image_root: str,
        processor,
        question_mapper_path: str = "data/question_mapper.json",
        llm_backend: str = "gpt",
        gpt_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        vllm_model: str = "Qwen3-14B",
        sample_n: int = 0,
        max_seq_len: int = 2048,
        image_size: Optional[int] = 448,
        seed: int = 42,
        special_token_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        no_mcq_postfix: bool = False,
    ):
        super().__init__()
        assert llm_backend in ("gpt", "vllm")
        self.image_root = image_root
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.llm_backend = llm_backend
        self.special_token_config = special_token_config
        self.model_config = model_config
        self.no_mcq_postfix = no_mcq_postfix

        self.cache = QuestionMapperCache(question_mapper_path)
        print(f"[FramingDataset] Cache at {question_mapper_path}: {len(self.cache)} entries")

        with open(data_path, "r") as f:
            raw_data = json.load(f)

        self.samples = self._filter_samples(raw_data)

        if sample_n > 0 and sample_n < len(self.samples):
            cached_samples = []
            uncached_samples = []
            for sample in self.samples:
                cached = self.cache.get(sample["id"])
                if cached is not None and "short_answer" in cached and "open_ended" in cached:
                    cached_samples.append(sample)
                else:
                    uncached_samples.append(sample)

            rng = random.Random(seed)
            if len(cached_samples) >= sample_n:
                selected_samples = rng.sample(cached_samples, sample_n)
            else:
                selected_samples = cached_samples.copy()
                remaining_needed = sample_n - len(selected_samples)
                if remaining_needed > 0 and len(uncached_samples) > 0:
                    selected_samples.extend(rng.sample(uncached_samples, min(remaining_needed, len(uncached_samples))))

            self.samples = selected_samples

        print(f"[FramingDataset] Loaded {len(self.samples)} samples from {data_path}")

        if OpenAI is None:
            self.llm_client = None
            self.llm_model = None
            return

        if llm_backend == "gpt":
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("llm_backend='gpt' requires an API key. Pass --openai_api_key or set OPENAI_API_KEY env var.")
            self.llm_client = OpenAI(api_key=api_key)
            self.llm_model = gpt_model
        else:
            try:
                self.llm_client = OpenAI(base_url=f"http://{vllm_host}:{vllm_port}/v1", api_key="EMPTY")
                self.llm_model = vllm_model
            except Exception as e:
                print(f"[FramingDataset] Could not create vLLM client: {e}")
                self.llm_client = None
                self.llm_model = None

    @staticmethod
    def _filter_samples(raw_data: List[Dict]) -> List[Dict]:
        """Keep only single-image, first-turn QA pairs."""
        filtered = []
        for item in raw_data:
            if item.get("image", None) is None:
                continue
            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue
            human_turn = convs[0]
            gpt_turn = convs[1]
            if human_turn.get("from") != "human" or gpt_turn.get("from") != "gpt":
                continue
            question = human_turn.get("value", "")
            answer = gpt_turn.get("value", "")
            if not question or not answer or len(answer.split()) > 16:
                continue
            question_clean = question.replace("<image>", "").strip()
            if not question_clean:
                continue
            filtered.append({
                "id": str(item.get("id", len(filtered))),
                "image": item["image"],
                "question": question_clean,
                "answer": answer,
            })
        return filtered

    def _load_image(self, image_rel_path: str) -> Image.Image:
        image_path = os.path.join(self.image_root, image_rel_path)
        img = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            w, h = img.size
            longest_side = max(w, h)
            if longest_side > self.image_size:
                if w > h:
                    new_w = self.image_size
                    new_h = int(h * self.image_size / w)
                else:
                    new_h = self.image_size
                    new_w = int(w * self.image_size / h)
                img = img.resize((new_w, new_h), Image.BILINEAR)
        return img

    def _get_or_generate_framings(self, sample_id: str, question: str, answer: str) -> Dict[str, Any]:
        cached = self.cache.get(sample_id)
        if cached is not None and "short_answer" in cached and "open_ended" in cached:
            return cached
        entry = _generate_framings(self.llm_client, self.llm_model, question, answer, self.llm_backend)
        self.cache.set(sample_id, entry)
        return entry

    @staticmethod
    def _format_mcq_question(mcq_data: Dict[str, Any]) -> Tuple[str, str]:
        q = mcq_data["question"]
        options = mcq_data["options"]
        lines = [q] + [f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]
        return "\n".join(lines), mcq_data["answer_letter"]

    def _encode_sample(self, image: Image.Image, question: str, answer: str,
                       framing_idx: Optional[int] = None, instruction_postfix: str = "") -> Dict[str, torch.Tensor]:
        """Tokenize a (image, question, answer) sample using the Qwen2.5-VL processor."""
        if framing_idx is not None and self.special_token_config is not None:
            prefix_text = self.special_token_config["prefix_texts"][framing_idx]
            postfix_text = self.special_token_config["postfix_texts"][framing_idx]

            if self.special_token_config["position"] == "prefix":
                question = prefix_text + " " + question
            elif self.special_token_config["position"] == "postfix":
                question = question + " " + postfix_text
            elif self.special_token_config["position"] == "both":
                question = prefix_text + " " + question + " " + postfix_text
            elif self.special_token_config["position"] == "infix":
                question = question + " " + prefix_text + instruction_postfix

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + answer + self.processor.tokenizer.eos_token

        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt", padding=False)

        answer_suffix = answer + self.processor.tokenizer.eos_token
        answer_ids = self.processor.tokenizer(answer_suffix, return_tensors="pt", add_special_tokens=False).input_ids[0]

        prompt_ids = inputs.input_ids[0]
        full_input_ids = torch.cat([prompt_ids, answer_ids], dim=0)

        if full_input_ids.shape[0] > self.max_seq_len:
            full_input_ids = full_input_ids[:self.max_seq_len]

        seq_len = full_input_ids.shape[0]
        full_attention_mask = torch.ones(seq_len, dtype=torch.long)

        IMAGE_TOKEN_ID = 151655  # Qwen2.5-VL image token
        img_mask = full_input_ids == IMAGE_TOKEN_ID
        img_indices = img_mask.nonzero(as_tuple=False).squeeze(-1)
        if img_indices.numel() > 0:
            img_start = img_indices[0].item()
            img_end = img_indices[-1].item() + 1
        else:
            img_start = 0
            img_end = 0

        prompt_len = prompt_ids.shape[0]
        answer_start = min(prompt_len, seq_len - 1)
        answer_end = seq_len

        result = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "answer_start": answer_start,
            "answer_end": answer_end,
            "img_start": img_start,
            "img_end": img_end,
        }

        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            result["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
            result["image_grid_thw"] = inputs.image_grid_thw[0]

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        question = sample["question"]
        answer = sample["answer"]
        sample_id = sample["id"]

        framings = self._get_or_generate_framings(sample_id, question, answer)

        open_ended_data = framings["open_ended"]
        yesno_data = framings["yesno"]
        mcq_data = framings["mcq"]
        short_answer = framings["short_answer"]
        mcq_question, mcq_answer = self._format_mcq_question(mcq_data)

        yesno_postfix = random.choice(YESNO_POSTFIXES)
        mcq_postfix = "" if self.no_mcq_postfix else random.choice(MCQ_POSTFIXES)
        open_c_postfix = random.choice(OPEN_CONSTRAINED_POSTFIXES)

        if self.special_token_config is not None and self.special_token_config["position"] == "infix":
            open_enc = self._encode_sample(image, question, answer, framing_idx=None, instruction_postfix="")
            open_c_enc = self._encode_sample(image, open_ended_data["question"] + open_c_postfix, short_answer, framing_idx=None, instruction_postfix="")
            yesno_enc = self._encode_sample(image, yesno_data["question"], yesno_data["answer"], framing_idx=1, instruction_postfix=yesno_postfix)
            mcq_enc = self._encode_sample(image, mcq_question, mcq_answer, framing_idx=2, instruction_postfix=mcq_postfix)
        else:
            open_enc = self._encode_sample(image, question, answer, framing_idx=None)
            open_c_enc = self._encode_sample(image, open_ended_data["question"] + open_c_postfix, short_answer, framing_idx=None)
            yesno_enc = self._encode_sample(image, yesno_data["question"] + yesno_postfix, yesno_data["answer"], framing_idx=1)
            mcq_enc = self._encode_sample(image, mcq_question + mcq_postfix, mcq_answer, framing_idx=2)

        image_path = os.path.join(self.image_root, sample["image"])
        for enc, q_text, a_text in [
            (open_enc, question, answer),
            (open_c_enc, open_ended_data["question"] + open_c_postfix, short_answer),
            (yesno_enc, yesno_data["question"] + yesno_postfix, yesno_data["answer"]),
            (mcq_enc, mcq_question + mcq_postfix, mcq_answer),
        ]:
            enc["image_path"] = image_path
            enc["question_text"] = q_text
            enc["answer_text"] = a_text
            enc["sample_id"] = sample_id

        return {
            "image": image,
            "open": open_enc,
            "open_constrained": open_c_enc,
            "yesno": yesno_enc,
            "mcq": mcq_enc,
            "sample_id": sample_id,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def framing_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of quadruplets into flat tensors."""
    def _pad_and_stack(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        max_len = max(s["input_ids"].shape[0] for s in samples)
        B = len(samples)

        input_ids = torch.full((B, max_len), fill_value=0, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        answer_starts, answer_ends = [], []
        img_starts, img_ends = [], []

        for i, s in enumerate(samples):
            L = s["input_ids"].shape[0]
            input_ids[i, :L] = s["input_ids"]
            attention_mask[i, :L] = s["attention_mask"]
            answer_starts.append(s["answer_start"])
            answer_ends.append(s["answer_end"])
            img_starts.append(s["img_start"])
            img_ends.append(s["img_end"])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_starts": answer_starts,
            "answer_ends": answer_ends,
            "img_starts": img_starts,
            "img_ends": img_ends,
        }

        if all("pixel_values" in s for s in samples):
            result["pixel_values"] = [s["pixel_values"] for s in samples]
            if all("image_grid_thw" in s for s in samples):
                result["image_grid_thw"] = torch.stack([s["image_grid_thw"] for s in samples])

        for str_key in ("image_path", "question_text", "answer_text", "sample_id"):
            if str_key in samples[0]:
                result[str_key] = [s[str_key] for s in samples]

        return result

    return {
        "open": _pad_and_stack([item["open"] for item in batch]),
        "open_constrained": _pad_and_stack([item["open_constrained"] for item in batch]),
        "yesno": _pad_and_stack([item["yesno"] for item in batch]),
        "mcq": _pad_and_stack([item["mcq"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
    }
