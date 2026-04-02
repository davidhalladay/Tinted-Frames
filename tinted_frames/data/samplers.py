"""
Dataset Samplers for VQA Benchmarks

Provides sampling strategies for different VQA datasets with support for:
- Random sampling
- Balanced sampling (by task type, grid position, etc.)
- Filtering by answer type, structural type, etc.
"""

import json
import random
import os
import csv
import sys
import base64
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

MCQ_POSTFIX = " Please answer directly with only the letter of the correct option and nothing else."
YESNO_POSTFIX = " Please answer yes or no."
OPEN_ENDED_POSTFIX = " Please try to answer the question with short words or phrases if possible."

class BaseSampler:
    """Base class for dataset samplers."""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from the dataset.
        
        Args:
            n_samples: Number of samples to select
            
        Returns:
            List of sampled data dictionaries
        """
        raise NotImplementedError


class GQASampler(BaseSampler):
    """
    Sampler for GQA dataset with support for:
    - Open-form questions (query type)
    - Yes/No questions (logical type)
    - Scene graph / bounding box annotations
    """
    
    def __init__(
        self,
        questions_path: str,
        image_root: str,
        scene_graphs_path: Optional[str] = None,
        seed: int = 42,
        auto_remove_potential_binary_Q: bool = False
    ):
        """
        Args:
            questions_path: Path to GQA questions JSON file
            image_root: Root directory for GQA images
            scene_graphs_path: Optional path to scene graphs JSON
            seed: Random seed
            auto_remove_potential_binary_Q: If True, filter out questions that are
                likely binary (yes/no) based on question wording and answer content
        """
        super().__init__(seed)
        self.questions_path = questions_path
        self.image_root = image_root
        self.scene_graphs_path = scene_graphs_path
        self.auto_remove_potential_binary_Q = auto_remove_potential_binary_Q
        self.questions = self._load_questions()
        self.scene_graphs = self._load_scene_graphs() if scene_graphs_path else None
    
    def _load_questions(self) -> Dict[str, Any]:
        """Load GQA questions."""
        print(f"Loading GQA questions from: {self.questions_path}")
        with open(self.questions_path, 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} GQA questions")
        return questions
    
    def _load_scene_graphs(self) -> Dict[str, Any]:
        """Load GQA scene graphs."""
        print(f"Loading GQA scene graphs from: {self.scene_graphs_path}")
        with open(self.scene_graphs_path, 'r') as f:
            scene_graphs = json.load(f)
        print(f"Loaded scene graphs for {len(scene_graphs)} images")
        return scene_graphs
    
    def _is_potential_binary(self, question: str, answer: str) -> bool:
        """Return True if the question/answer pair looks like a binary (yes/no) question."""
        if " or " in question.lower():
            return True
        if "yes" in answer.lower() or "no" in answer.lower():
            return True
        if question.split(" ")[0].lower() in ("is", "are"):
            return True
        return False

    def sample_by_structural_type(
        self,
        structural_type: str,
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Sample questions by structural type.
        
        Args:
            structural_type: 'query' (open-form) or 'logical' (yes/no)
            n_samples: Number of samples

        Returns:
            List of sampled questions
        """
        # Filter by structural type
        filtered = []
        for qid, qdata in self.questions.items():
            if qdata['types']['structural'] == structural_type:
                if self.auto_remove_potential_binary_Q and self._is_potential_binary(
                    qdata['question'], qdata.get('answer', '')
                ):
                    continue
                filtered.append((qid, qdata))
        
        print(f"Found {len(filtered)} questions with structural_type='{structural_type}'")
        
        # Sample
        sampled = random.sample(filtered, min(n_samples, len(filtered)))
        
        # Format
        formatted_samples = []
        task_type = 'open_form' if structural_type == 'query' else 'yes_no'
        
        for qid, qdata in sampled:
            sample = {
                'task_type': task_type,
                'dataset': 'GQA',
                'question_id': qid,
                'image': os.path.join(self.image_root, qdata['imageId'] + '.jpg'),
                'text': qdata['question'] + OPEN_ENDED_POSTFIX,  # Standardize to 'text' for consistency
                'text_nopostfix': qdata['question'],
                'answer': qdata.get('answer', ''),
                'structural_type': qdata['types']['structural'],
                'detailed_type': qdata['types'].get('detailed', ''),
                'options': ['yes', 'no'] if task_type == 'yes_no' else []
            }

            # Add bounding box if available
            if self.scene_graphs and 'imageId' in qdata:
                image_id = qdata['imageId']
                if image_id in self.scene_graphs:
                    # Extract bounding box from annotations
                    bbox = self._extract_bbox_from_annotations(qdata, self.scene_graphs[image_id])
                    if bbox:
                        sample['bbox'] = bbox
            
            formatted_samples.append(sample)
        
        print(f"Sampled {len(formatted_samples)} GQA {task_type} questions")
        return formatted_samples
    
    def _extract_bbox_from_annotations(
        self,
        question_data: Dict[str, Any],
        scene_graph: Dict[str, Any]
    ) -> Optional[Dict[str, int]]:
        """
        Extract bounding box from question annotations using scene graph.
        
        Returns:
            Dict with keys: x, y, w, h or None
        """
        # Try to get object IDs from annotations
        annotations = question_data.get('annotations', {})
        
        # Look for objects in question or answer
        for key in ['question', 'answer']:
            if key in annotations:
                for obj_id in annotations[key].values():
                    if obj_id in scene_graph.get('objects', {}):
                        obj = scene_graph['objects'][obj_id]
                        if all(k in obj for k in ['x', 'y', 'w', 'h']):
                            return [obj['x'], obj['y'], obj['w'], obj['h']]
        
        return None
    
    def sample_randomly(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Randomly sample n_samples from GQA dataset without filtering.
        
        Args:
            n_samples: Number of samples to randomly select
            
        Returns:
            List of randomly sampled questions
        """
        # Convert questions dict to list
        all_questions = list(self.questions.items())
        
        if n_samples == -1:
            n_samples = len(all_questions)

        # Random sample
        sampled = random.sample(all_questions, min(n_samples, len(all_questions)))

        # Format
        formatted_samples = []

        for qid, qdata in sampled:
            if self.auto_remove_potential_binary_Q and self._is_potential_binary(
                qdata['question'], qdata.get('answer', '')
            ):
                continue

            structural_type = qdata['types']['structural']
            task_type = 'open_form' if structural_type == 'query' else 'yes_no'

            sample = {
                'task_type': task_type,
                'dataset': 'GQA',
                'question_id': qid,
                'image': os.path.join(self.image_root, qdata['imageId'] + '.jpg'),
                'text': qdata['question'] + OPEN_ENDED_POSTFIX,
                'text_nopostfix': qdata['question'],
                'answer': qdata.get('answer', ''),
                'structural_type': structural_type,
                'detailed_type': qdata['types'].get('detailed', ''),
                'options': ['yes', 'no'] if task_type == 'yes_no' else []
            }

            # Add bounding box if available
            if self.scene_graphs and 'imageId' in qdata:
                image_id = qdata['imageId']
                if image_id in self.scene_graphs:
                    bbox = self._extract_bbox_from_annotations(qdata, self.scene_graphs[image_id])
                    if bbox:
                        sample['bbox'] = bbox

            formatted_samples.append(sample)
        
        print(f"Randomly sampled {len(formatted_samples)} GQA questions")
        return formatted_samples
    
    def sample(
        self,
        n_open_form: int = 0,
        n_yes_no: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Sample both open-form and yes/no questions.
        
        Args:
            n_open_form: Number of open-form samples
            n_yes_no: Number of yes/no samples
            
        Returns:
            Combined list of samples
        """
        samples = []
        
        if n_open_form > 0:
            samples.extend(self.sample_by_structural_type('query', n_open_form))
        
        if n_yes_no > 0:
            samples.extend(self.sample_by_structural_type('logical', n_yes_no))
        
        return samples


class ReframedGQASampler:
    """
    Sampler for Reframed GQA dataset.
    Merges 'Original' and 'Reframed' questions into three unified pools:
    1. Yes/No
    2. MCQ
    3. Open-ended
    
    The 'Original' question is assigned to either Yes/No or Open-ended 
    based on whether the answer is 'yes'/'no' or a text phrase.
    """
    
    def __init__(
        self,
        questions_path: str,
        reframed_questions_path: str,
        image_root: str,
        scene_graphs_path: Optional[str] = None,
        seed: int = 42,
        auto_remove_potential_binary_Q: bool = True
    ):
        """
        Args:
            questions_path: Path to the reframed GQA JSON file
            reframed_questions_path: Path to the reframed GQA JSON file
            image_root: Root directory for GQA images
            scene_graphs_path: Optional path to scene graphs JSON (for bounding boxes)
            seed: Random seed
        """
        random.seed(seed)
        self.questions_path = questions_path
        self.reframed_questions_path = reframed_questions_path
        self.image_root = image_root
        self.scene_graphs_path = scene_graphs_path
        
        # Load raw data
        self.raw_data = self._load_json(self.questions_path)
        self.reframed_data = self._load_json(self.reframed_questions_path)
        self.scene_graphs = self._load_json(self.scene_graphs_path) if scene_graphs_path else None
        self.auto_remove_potential_binary_Q = auto_remove_potential_binary_Q
        
        # Three unified pools
        self.pools = {
            'yes_no': [],
            'mcq': [],
            'open_ended': []
        }
        self._organize_samples()

    def _load_json(self, path: str) -> Dict[str, Any]:
        print(f"Loading data from: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _organize_samples(self):
        """
        Iterates through the dataset and buckets every question (reframed AND original)
        into one of the three task pools.
        """
        print("Organizing samples into unified categories...")
        
        for qid, content in self.reframed_data.items():
            original = content.get('original', {})
            reframed = content.get('reframed', {})

            if self.auto_remove_potential_binary_Q:
                if 'open_ended' in reframed:
                    tmp_question = reframed['open_ended']['question']
                    tmp_answer = reframed['open_ended']['answer']
                else:
                    tmp_question = original['question']
                    tmp_answer = original['answer']
                flag = False
                if " or " in tmp_question.lower():
                    flag = True
                if "yes" in tmp_answer.lower() or "no" in tmp_answer.lower():
                    flag = True
                if "is" == tmp_question.split(" ")[0].lower() or "are" == tmp_question.split(" ")[0].lower():
                    flag = True
                if flag:
                    continue
            
            # Common Metadata
            image_id = original.get('imageId')
            types = original.get('types', {})
            structural_type = types.get('structural', 'unknown')
            detailed_type = types.get('detailed', 'unknown')
            annotations = self.raw_data[qid].get('annotations', {})
            original['annotations'] = annotations  # Keep for bbox extraction

            # -------------------------------------------------
            # 1. PROCESS REFRAMED QUESTIONS
            # -------------------------------------------------
            
            # Reframed: Yes/No
            if 'yes_no' in reframed:
                item = reframed['yes_no']
                self.pools['yes_no'].append({
                    'qid': qid,
                    'image_id': image_id,
                    'original_data': original, # Kept for bbox extraction
                    'question': item['question'],
                    'answer': item['answer'],
                    'structural': structural_type,
                    'detailed': detailed_type,
                    'formatted_text': item['question'] + YESNO_POSTFIX,
                    'formatted_text_nopostfix': item['question'],
                    'options': ['yes', 'no']
                })

            # Reframed: Open Ended
            if 'open_ended' in reframed:
                item = reframed['open_ended']
                self.pools['open_ended'].append({
                    'qid': qid,
                    'image_id': image_id,
                    'original_data': original,
                    'question': item['question'],
                    'answer': item['answer'],
                    'structural': structural_type,
                    'detailed': detailed_type,
                    'formatted_text': item['question'] + OPEN_ENDED_POSTFIX,
                    'formatted_text_nopostfix': item['question'],
                    'options': []
                })

            # Reframed: MCQ
            if 'mcq' in reframed:
                item = reframed['mcq']
                options = item.get('options', [])
                # Format options: (A) Option 1...
                # shuffle options to avoid positional bias
                random.shuffle(options)
                
                options_text = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                full_text = f"{item['question']}"
                self.pools['mcq'].append({
                    'qid': qid,
                    'image_id': image_id,
                    'original_data': original,
                    'question': item['question'],
                    'answer': f"{chr(65+options.index(item.get('answer_text', '')))}",
                    'structural': structural_type,
                    'detailed': detailed_type,
                    'formatted_text': full_text + f"\nOptions:\n{options_text}" + MCQ_POSTFIX,
                    'formatted_text_nopostfix': full_text + f"\nOptions:\n{options_text}",
                    'options': options
                })

            # -------------------------------------------------
            # 2. PROCESS ORIGINAL QUESTION
            # -------------------------------------------------
            # Determine framing based on whether answer is yes/no
            
            original_text = original['question']
            original_ans = original['answer']
            ans_clean = str(original_ans).lower().strip()
            
            if ans_clean in ['yes', 'no']:
                # Assign to Yes/No pool
                self.pools['yes_no'].append({
                    'qid': qid,
                    'image_id': image_id,
                    'original_data': original,
                    'question': original_text,
                    'answer': original_ans,
                    'structural': structural_type,
                    'detailed': detailed_type,
                    'formatted_text': original_text + YESNO_POSTFIX,
                    'formatted_text_nopostfix': original_text,
                    'options': ['yes', 'no']
                })
            else:
                # Assign to Open-ended pool
                self.pools['open_ended'].append({
                    'qid': qid,
                    'image_id': image_id,
                    'original_data': original,
                    'question': original_text,
                    'answer': original_ans,
                    'structural': structural_type,
                    'detailed': detailed_type,
                    'formatted_text': original_text + OPEN_ENDED_POSTFIX,
                    'formatted_text_nopostfix': original_text,
                    'options': []
                })

        print(f"Pool Sizes -> Yes/No: {len(self.pools['yes_no'])}, MCQ: {len(self.pools['mcq'])}, Open: {len(self.pools['open_ended'])}")

    def _extract_bbox_from_annotations(
        self,
        original_data: Dict[str, Any],
        scene_graph: Dict[str, Any]
    ) -> Optional[Dict[str, int]]:
        """Extract bounding box using annotations from the original data structure."""
        annotations = original_data.get('annotations', {})
        
        # We search both 'question' and 'answer' annotations for object IDs
        # Note: Even for reframed questions, we use the original annotations 
        # as the underlying object references (semantic ID) remain valid.
        all_bboxes = []
        for key in ['question', 'answer']:
            if key in annotations:
                for obj_id in annotations[key].values():
                    # Validate object exists in scene graph
                    if obj_id in scene_graph.get('objects', {}):
                        obj = scene_graph['objects'][obj_id]
                        if all(k in obj for k in ['x', 'y', 'w', 'h']):
                            all_bboxes.append({
                                'x': obj['x'],
                                'y': obj['y'],
                                'w': obj['w'],
                                'h': obj['h']
                            })
        return all_bboxes

    def _format_sample(self, raw_sample: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Convert internal dict to final output format."""
        
        output = {
            'task_type': task_type,
            'dataset': 'GQA',
            'question_id': raw_sample['qid'],
            'image': os.path.join(self.image_root, raw_sample['image_id'] + '.jpg'),
            'text': raw_sample['formatted_text'],
            'text_nopostfix': raw_sample['formatted_text_nopostfix'],
            'answer': raw_sample['answer'],
            'structural_type': raw_sample['structural'],
            'detailed_type': raw_sample['detailed'],
            'options': raw_sample.get('options', [])
        }

        # Add bounding box if scene graphs are available
        if self.scene_graphs and raw_sample['image_id'] in self.scene_graphs:
            bbox = self._extract_bbox_from_annotations(
                raw_sample['original_data'], 
                self.scene_graphs[raw_sample['image_id']]
            )
            if bbox:
                output['bbox'] = bbox[0]
                output['all_bboxes'] = bbox  # In case multiple bboxes are found
                
        return output

    def sample(self, n_samples: int, single_obj_only: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Samples `n_samples` for EACH framing type, using the same random indices across all task types.
        
        Args:
            n_samples: The number of samples to retrieve for each category.
            single_obj_only: Whether to filter to single-object samples only.
            yes_only: Whether to sample only yes answers to yes/no questions.
            
        Returns:
            Tuple containing three lists: (yes_no, mcq, open_ended)
        """
        # prepare pools_w_bbox 
        task_types = self.pools.keys()
        pools_w_bbox = dict()
        for task in task_types:
            pool = self.pools[task]
            pools_w_bbox[task] = [self._format_sample(s, task) for s in pool]
        self.pools = pools_w_bbox

        # remove the sample in self.pools[task] if more than one object in boxes
        if single_obj_only:
            print("Filtering to single-object samples only...")
            print("Number of samples before filtering:")
            for task in task_types:
                print(f"  {task}: {len(self.pools[task])}")
            pools_w_single_obj = dict()
            for task in task_types:
                pool = self.pools[task]
                pools_w_single_obj[task] = [s for s in pool if len(s['all_bboxes']) == 1]
            self.pools = pools_w_single_obj
            print("Number of samples after filtering:")
            for task in task_types:
                print(f"  {task}: {len(self.pools[task])}")

        results = {}
        task_types = ['yes_no', 'mcq', 'open_ended']

        # Find the minimum pool size across all task types
        min_pool_size = min(len(self.pools[task]) for task in task_types)
        count = min(n_samples, min_pool_size)

        # If any pool is empty, return empty lists for all
        if count == 0:
            for task in task_types:
                results[task] = []
            print(f"Sampled: 0 Yes/No, 0 MCQ, 0 Open-ended (one or more pools empty)")
            return results['yes_no'], results['mcq'], results['open_ended']
        
        # Generate shared random indices
        indices = random.sample(range(min_pool_size), count)
        for task in task_types:
            pool = self.pools[task]
            # Use the same indices for all pools (truncate pool if needed)
            results[task] = [pool[i] for i in indices]
            # results[task] = [self._format_sample(s, task) for s in selected_raw]

        print(f"Sampled: {len(results['yes_no'])} Yes/No, {len(results['mcq'])} MCQ, {len(results['open_ended'])} Open-ended (shared random selection)")
        return results['yes_no'], results['mcq'], results['open_ended']


class SeedBenchSampler(BaseSampler):
    """
    Sampler for SeedBench dataset (Multiple-choice questions).
    Loads directly from SEED-Bench.json without requiring a pre-processed JSONL file.
    """

    QUESTION_TYPE_MAPPING = {
        1: "Scene Understanding",
        2: "Instance Identity",
        3: "Instance Attributes",
        4: "Instance Location",
        5: "Instances Counting",
        6: "Spatial Relation",
        7: "Instance Interaction",
        8: "Visual Reasoning",
        9: "Text Understanding",
        10: "Action Recognition",
        11: "Action Prediction",
        12: "Procedure Understanding"
    }

    def __init__(
        self,
        answers_json_path: str,
        image_root: str,
        seed: int = 42
    ):
        """
        Args:
            answers_json_path: Path to SEED-Bench.json file containing questions and answers
            image_root: Root directory for SeedBench images
            seed: Random seed
        """
        super().__init__(seed)
        self.answers_json_path = answers_json_path
        self.image_root = image_root
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load SeedBench data directly from SEED-Bench.json."""
        print(f"Loading SeedBench data from: {self.answers_json_path}")
        with open(self.answers_json_path, 'r') as f:
            answers_data = json.load(f)

        all_data = []
        for question in answers_data.get('questions', []):
            # Skip video samples
            if question.get('data_type', 'image') != 'image':
                continue

            data_id = question.get('data_id', '')
            image_path = os.path.join(self.image_root, data_id)
            question_type_id = question.get('question_type_id', 0)
            category = self.QUESTION_TYPE_MAPPING.get(question_type_id, 'unknown')
            options = [
                question.get('choice_a', ''),
                question.get('choice_b', ''),
                question.get('choice_c', ''),
                question.get('choice_d', ''),
            ]

            all_data.append({
                'question_id': str(question.get('question_id', '')),
                'image': image_path,
                'question': question.get('question', ''),
                'answer': question.get('answer', ''),
                'options': options,
                'category': category,
                'question_type_id': question_type_id,
            })

        print(f"Loaded {len(all_data)} SeedBench image samples")
        return all_data

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from SeedBench.

        Returns:
            List of samples with 'task_type': 'mcq'
        """
        if n_samples == -1:
            n_samples = len(self.data)
        sampled = random.sample(self.data, min(n_samples, len(self.data)))

        formatted_samples = []
        for sample in sampled:
            options = sample['options']
            mcq_text = sample['question'] + "\n"
            for i, opt in enumerate(options):
                mcq_text += f"{chr(65+i)}. {opt}\n"
            mcq_text = mcq_text.strip()

            answer_letter = sample['answer']
            answer_idx = ord(answer_letter.upper()) - ord('A')
            answer_text = options[answer_idx] if 0 <= answer_idx < len(options) else ''

            formatted_samples.append({
                'task_type': 'mcq',
                'dataset': 'SeedBench',
                'question_id': sample['question_id'],
                'image': sample['image'],
                'text': mcq_text + MCQ_POSTFIX,
                'text_nopostfix': mcq_text,
                'question': sample['question'],
                'category': sample['category'],
                'answer': answer_letter,
                'answer_text': answer_text,
                'options': options,
            })

        print(f"Sampled {len(formatted_samples)} SeedBench MCQ questions")
        return formatted_samples

    def sample_balanced(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples with balanced representation across question types 1-9.
        (Types 10-12 are video tasks and are already excluded by _load_data.)

        Each image question type gets as close to an equal share as possible.
        Remainder samples are distributed one-by-one to the smallest types first.

        Args:
            n_samples: Total number of samples to select. -1 means all data.

        Returns:
            List of MCQ-formatted samples.
        """
        if n_samples == -1:
            n_samples = len(self.data)

        # Group by question_type_id (only image types 1-9 present after _load_data)
        by_type: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in self.data:
            by_type[item['question_type_id']].append(item)

        type_ids_sorted = sorted(by_type.keys())
        n_types = len(type_ids_sorted)
        per_type = n_samples // n_types if n_types > 0 else 0
        remainder = n_samples - per_type * n_types

        sampled = []
        for i, type_id in enumerate(type_ids_sorted):
            available = by_type[type_id]
            n_take = per_type + (1 if i < remainder else 0)
            n_take = min(n_take, len(available))
            sampled.extend(random.sample(available, n_take))

        random.shuffle(sampled)

        # Print distribution
        print("Balanced sampling distribution (SeedBench):")
        for type_id in type_ids_sorted:
            type_name = self.QUESTION_TYPE_MAPPING.get(type_id, 'unknown')
            count = sum(1 for s in sampled if s['question_type_id'] == type_id)
            print(f"  Type {type_id} ({type_name}): {count}")

        formatted_samples = []
        for sample in sampled:
            options = sample['options']
            mcq_text = sample['question'] + "\n"
            for i, opt in enumerate(options):
                mcq_text += f"{chr(65+i)}. {opt}\n"
            mcq_text = mcq_text.strip()

            answer_letter = sample['answer']
            answer_idx = ord(answer_letter.upper()) - ord('A')
            answer_text = options[answer_idx] if 0 <= answer_idx < len(options) else ''

            formatted_samples.append({
                'task_type': 'mcq',
                'dataset': 'SeedBench',
                'question_id': sample['question_id'],
                'image': sample['image'],
                'text': mcq_text + MCQ_POSTFIX,
                'text_nopostfix': mcq_text,
                'question': sample['question'],
                'category': sample['category'],
                'answer': answer_letter,
                'answer_text': answer_text,
                'options': options,
            })

        print(f"Sampled {len(formatted_samples)} SeedBench MCQ questions (balanced)")
        return formatted_samples


class SeedBenchUnifiedSampler(BaseSampler):
    """
    Unified sampler for SeedBench-YesNo dataset with support for three modes:
    - yes_no: Yes/No questions
    - mcq: Multiple-choice questions
    - open_ended: Open-ended questions
    """
    
    # Question type mapping from numbers to names
    QUESTION_TYPE_MAPPING = {
        1: "Scene Understanding",
        2: "Instance Identity",
        3: "Instance Attributes",
        4: "Instance Location",
        5: "Instances Counting",
        6: "Spatial Relation",
        7: "Instance Interaction",
        8: "Visual Reasoning",
        9: "Text Understanding",
        10: "Action Recognition",
        11: "Action Prediction",
        12: "Procedure Understanding"
    }
    
    def __init__(
        self,
        json_path: str,
        image_root: str,
        seed: int = 42
    ):
        """
        Args:
            json_path: Path to SeedBench-YesNo JSON file
            image_root: Root directory for SeedBench images
            seed: Random seed
        """
        super().__init__(seed)
        self.json_path = json_path
        self.image_root = image_root
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load SeedBench-YesNo data from JSON."""
        print(f"Loading SeedBench-YesNo data from: {self.json_path}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        questions = data.get('questions', [])
        print(f"Loaded {len(questions)} SeedBench-YesNo samples")
        return questions
    
    def _format_as_yes_no(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample as yes/no question."""
        question_text = sample['question'] + "\nPlease return yes or no."
        
        # Convert question_type number to name
        question_type_num = sample.get('question_type_id', 0)
        category = self.QUESTION_TYPE_MAPPING.get(question_type_num, 'unknown')
        
        return {
            'task_type': 'yes_no',
            'dataset': 'SeedBench-YesNo',
            'question_id': sample['question_id'],
            'image': os.path.join(self.image_root, sample['data_id']),
            'text': question_text,
            'answer': sample['answer'],
            'category': category,
            'options': ['yes', 'no']
        }

    def _format_as_mcq(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample as multiple-choice question using original choices."""
        # Get original question and choices
        question = sample['original_question']
        choices = [
            sample.get('choice_a', ''),
            sample.get('choice_b', ''),
            sample.get('choice_c', ''),
            sample.get('choice_d', '')
        ]
        
        # Filter out empty choices
        choices = [c for c in choices if c]
        
        # Format question with options
        formatted_text = question + "\n"
        for idx, choice in enumerate(choices):
            letter = chr(65 + idx)  # A, B, C, D
            formatted_text += f"{letter}. {choice}\n"
        formatted_text += "Return the letter only."
        
        # Convert question_type number to name
        question_type_num = sample.get('question_type_id', 0)
        category = self.QUESTION_TYPE_MAPPING.get(question_type_num, 'unknown')
        
        return {
            'task_type': 'mcq',
            'dataset': 'SeedBench-YesNo',
            'question_id': sample['question_id'],
            'image': os.path.join(self.image_root, sample['data_id']),
            'text': formatted_text,
            'answer': sample.get('original_answer', ''),
            'category': category,
            'options': choices
        }

    def _format_as_open_ended(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample as open-ended question."""
        # Use original question without choices
        question = sample['original_question']
        question_text = question + "\nPlease return a word or a phrase."
        
        # Get the text of the correct answer choice
        original_answer = sample.get('original_answer', '')
        answer_text = ''
        if original_answer:
            choice_key = f'original_choice_{original_answer.lower()}'
            answer_text = sample.get(choice_key, '')
        
        # Convert question_type number to name
        question_type_num = sample.get('question_type_id', 0)
        category = self.QUESTION_TYPE_MAPPING.get(question_type_num, 'unknown')
        
        return {
            'task_type': 'open_form',
            'dataset': 'SeedBench-YesNo',
            'question_id': sample['question_id'],
            'image': os.path.join(self.image_root, sample['data_id']),
            'text': question_text,
            'answer': answer_text,
            'category': category,
            'options': []
        }

    def sample(
        self,
        n_samples: int,
        mode: str = 'yes_no'
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Sample n_samples from SeedBench-YesNo dataset.
        
        Args:
            n_samples: Number of samples to select
            mode: Question format mode ('yes_no', 'mcq', 'open_ended', or 'yes')
            
        Returns:
            Tuple of (formatted_samples, question_ids)
        """
        random.seed(self.seed)
        
        if mode not in ['yes_no', 'mcq', 'open_ended', 'yes']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'yes_no', 'mcq', 'open_ended', or 'yes'")
        
        # Filter data based on mode
        if mode == 'yes':
            # Only include samples where answer is 'yes'
            filtered_data = [s for s in self.data if s.get('answer', '').lower() == 'yes']
            print(f"Filtered to {len(filtered_data)} samples with answer='yes' (from {len(self.data)} total)")
            sampled = random.sample(filtered_data, min(n_samples, len(filtered_data)))
        else:
            # Sample from all data
            sampled = random.sample(self.data, min(n_samples, len(self.data)))
        
        # Format based on mode
        formatted_samples = []
        question_ids = []
        for sample in sampled:
            question_ids.append(sample['question_id'])
            if mode in ['yes_no', 'yes']:
                formatted_samples.append(self._format_as_yes_no(sample))
            elif mode == 'mcq':
                formatted_samples.append(self._format_as_mcq(sample))
            elif mode == 'open_ended':
                formatted_samples.append(self._format_as_open_ended(sample))
        
        print(f"Sampled {len(formatted_samples)} SeedBench-YesNo {mode} questions")
        return formatted_samples, question_ids

    def sample_task_equal(
        self,
        n_samples: int,
        mode: str = 'yes_no'
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Sample n_samples from SeedBench-YesNo dataset with equal representation
        per question type (task category).

        Args:
            n_samples: Total number of samples to select
            mode: Question format mode ('yes_no', 'mcq', 'open_ended', or 'yes')

        Returns:
            Tuple of (formatted_samples, question_ids)
        """
        random.seed(self.seed)

        if mode not in ['yes_no', 'mcq', 'open_ended', 'yes']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'yes_no', 'mcq', 'open_ended', or 'yes'")

        # Filter data if mode == 'yes'
        if mode == 'yes':
            pool = [s for s in self.data if s.get('answer', '').lower() == 'yes']
        else:
            pool = self.data

        # Group by question_type_id
        by_type = defaultdict(list)
        for s in pool:
            type_id = s.get('question_type_id', 0)
            by_type[type_id].append(s)

        # Calculate per-type budget
        n_types = len(by_type)
        per_type = n_samples // n_types if n_types > 0 else 0
        remainder = n_samples - per_type * n_types

        # Sample equally from each type
        sampled = []
        type_ids_sorted = sorted(by_type.keys())
        for i, type_id in enumerate(type_ids_sorted):
            available = by_type[type_id]
            # Distribute remainder to the first few types
            n_take = per_type + (1 if i < remainder else 0)
            n_take = min(n_take, len(available))
            sampled.extend(random.sample(available, n_take))

        random.shuffle(sampled)

        # Print distribution
        print(f"Task-equal sampling distribution:")
        for type_id in type_ids_sorted:
            type_name = self.QUESTION_TYPE_MAPPING.get(type_id, 'unknown')
            count = sum(1 for s in sampled if s.get('question_type_id', 0) == type_id)
            print(f"  {type_name}: {count}")

        # Format
        formatted_samples = []
        question_ids = []
        for sample in sampled:
            question_ids.append(sample['question_id'])
            if mode in ['yes_no', 'yes']:
                formatted_samples.append(self._format_as_yes_no(sample))
            elif mode == 'mcq':
                formatted_samples.append(self._format_as_mcq(sample))
            elif mode == 'open_ended':
                formatted_samples.append(self._format_as_open_ended(sample))

        print(f"Sampled {len(formatted_samples)} SeedBench-YesNo {mode} questions (task-equal)")
        return formatted_samples, question_ids

    def sample_by_qid(
        self,
        question_ids: List[str],
        mode: str = 'yes_no'
    ) -> List[Dict[str, Any]]:
        """
        Sample specific questions by question IDs.
        
        Args:
            question_ids: List of question IDs to sample
            mode: Question format mode ('yes_no', 'mcq', 'open_ended', or 'yes')
            
        Returns:
            List of formatted samples matching the question IDs
        """
        if mode not in ['yes_no', 'mcq', 'open_ended', 'yes']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'yes_no', 'mcq', 'open_ended', or 'yes'")
        
        # Create a lookup dict for fast access
        qid_to_sample = {sample['question_id']: sample for sample in self.data}
        
        # Find samples matching the provided question IDs
        formatted_samples = []
        missing_qids = []
        
        for qid in question_ids:
            if qid in qid_to_sample:
                sample = qid_to_sample[qid]
                
                # Format based on mode
                if mode in ['yes_no', 'yes']:
                    formatted_samples.append(self._format_as_yes_no(sample))
                elif mode == 'mcq':
                    formatted_samples.append(self._format_as_mcq(sample))
                elif mode == 'open_ended':
                    formatted_samples.append(self._format_as_open_ended(sample))
            else:
                missing_qids.append(qid)
        
        if missing_qids:
            print(f"Warning: {len(missing_qids)} question IDs not found in dataset")
        
        print(f"Sampled {len(formatted_samples)} SeedBench-YesNo {mode} questions by QID")
        return formatted_samples


class VstarSampler(BaseSampler):
    """
    Sampler for V-Star benchmark dataset.
    Loads data from four subdirectories: direct_attributes, GPT4V-hard, OCR, and relative_position.
    Each sample contains a question with multiple choice options (first option is ground truth).
    """
    
    def __init__(
        self,
        vstar_root: str,
        image_root: str = None,
        seed: int = 42
    ):
        """
        Args:
            vstar_root: Root directory for V-Star benchmark (contains subdirectories)
            image_root: Root directory for V-Star images. If None, uses vstar_root
            seed: Random seed
        """
        super().__init__(seed)
        self.vstar_root = vstar_root
        self.image_root = image_root if image_root else vstar_root
        self.categories = ['direct_attributes', 'GPT4V-hard', 'OCR', 'relative_position']
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load all V-Star data from all category subdirectories."""
        print(f"Loading V-Star data from: {self.vstar_root}")
        
        all_data = []
        
        for category in self.categories:
            category_path = os.path.join(self.vstar_root, category)
            
            if not os.path.exists(category_path):
                print(f"Warning: Category directory not found: {category_path}")
                continue
            
            print(f"Loading category: {category}")
            
            # Get all JSON files in this category
            json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
            
            for json_file in json_files:
                json_path = os.path.join(category_path, json_file)
                
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract the base filename (without extension) for image lookup
                    base_name = json_file.replace('.json', '')
                    
                    # Find the actual image file (handle both .jpg and .JPG)
                    image_filename = None
                    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', 'webp']:
                        candidate = f"{base_name}{ext}"
                        candidate_path = os.path.join(category_path, candidate)
                        if os.path.exists(candidate_path):
                            image_filename = candidate
                            break
                    
                    if not image_filename:
                        print(f"Warning: No image file found for {base_name} in {category}")
                        continue
                    
                    sample = {
                        'question_id': base_name,
                        'category': category,
                        'target_object': data.get('target_object', []),
                        'bbox': data.get('bbox', []),
                        'question': data.get('question', ''),
                        'options': data.get('options', []),
                        'image_filename': image_filename
                    }
                    
                    all_data.append(sample)
                
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to load {json_path}: {e}")
                    continue
        
        print(f"Loaded {len(all_data)} V-Star samples total")
        for category in self.categories:
            count = sum(1 for s in all_data if s['category'] == category)
            print(f"  {category}: {count}")
        
        return all_data
    
    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from V-Star dataset.
        
        Returns:
            List of samples with 'task_type': 'mcq' and matched output keys
        """
        if n_samples == -1:
            n_samples = len(self.data)
        
        sampled = random.sample(self.data, min(n_samples, len(self.data)))
        
        formatted_samples = []
        for sample in sampled:
            options = sample['options']
            
            # First option is the ground truth answer
            # Shuffle options and track which one is the correct answer
            shuffled_options = options.copy()
            random.shuffle(shuffled_options)
            
            # Find which position the correct answer (first option) is now at
            correct_answer = options[0]
            correct_idx = shuffled_options.index(correct_answer)
            correct_letter = chr(65 + correct_idx)  # A, B, C, D...
            
            # Format question with multiple choice options
            formatted_text = sample['question'] + "\n"
            for idx, option in enumerate(shuffled_options):
                letter = chr(65 + idx)
                formatted_text += f"{letter}. {option}\n"
            formatted_text_wpostfix = formatted_text + MCQ_POSTFIX
            
            # Get primary bbox (first one if multiple exist)
            bbox = None
            all_bboxes = None
            if sample['bbox'] and len(sample['bbox']) > 0:
                bbox = sample['bbox'][0]  # [x, y, width, height]
                all_bboxes = sample['bbox']
            
            formatted_sample = {
                'task_type': 'mcq',
                'dataset': 'V-Star',
                'question_id': sample['question_id'],
                'image': os.path.join(self.image_root, sample['category'], sample['image_filename']),
                'text': formatted_text_wpostfix,
                'question': sample.get('question', ''),
                'text_nopostfix': formatted_text,
                'answer': correct_letter,
                'answer_text': correct_answer,
                'category': sample['category'],
                'target_object': sample['target_object'],
                'bbox': bbox,
                'all_bboxes': all_bboxes,
                'options': shuffled_options
            }

            formatted_samples.append(formatted_sample)

        print(f"Sampled {len(formatted_samples)} V-Star MCQ questions")
        return formatted_samples
    
    def sample_by_category(
        self,
        n_samples_per_category: int
    ) -> List[Dict[str, Any]]:
        """
        Sample n_samples_per_category from each category, ensuring balanced distribution.
        
        Args:
            n_samples_per_category: Number of samples to select from each category
            
        Returns:
            Combined list of balanced samples across all categories
        """
        all_samples = []
        
        for category in self.categories:
            # Filter samples by category
            category_samples = [s for s in self.data if s['category'] == category]
            
            if not category_samples:
                print(f"Warning: No samples found for category: {category}")
                continue
            
            # Sample from this category
            n_take = min(n_samples_per_category, len(category_samples))
            sampled = random.sample(category_samples, n_take)
            
            # Format samples
            for sample in sampled:
                options = sample['options']
                shuffled_options = options.copy()
                random.shuffle(shuffled_options)
                
                correct_answer = options[0]
                correct_idx = shuffled_options.index(correct_answer)
                correct_letter = chr(65 + correct_idx)
                
                formatted_text = sample['question'] + "\n"
                for idx, option in enumerate(shuffled_options):
                    letter = chr(65 + idx)
                    formatted_text += f"{letter}. {option}\n"
                formatted_text += MCQ_POSTFIX
                
                bbox = None
                if sample['bbox'] and len(sample['bbox']) > 0:
                    bbox = sample['bbox'][0]
                
                formatted_sample = {
                    'task_type': 'mcq',
                    'dataset': 'V-Star',
                    'question_id': sample['question_id'],
                    'image': os.path.join(self.image_root, sample['category'], sample['image_filename']),
                    'text': formatted_text,
                    'text_nopostfix': sample['question'],
                    'answer': correct_letter,
                    'answer_text': correct_answer,
                    'category': sample['category'],
                    'target_object': sample['target_object'],
                    'bbox': bbox,
                    'options': shuffled_options
                }

                all_samples.append(formatted_sample)

        print(f"Sampled {len(all_samples)} V-Star MCQ questions ({n_samples_per_category} per category)")
        return all_samples


class VstarFramingSampler(BaseSampler):
    """
    Sampler for V-Star benchmark using pre-generated GPT reframings.
    Loads vstar_framing.json and returns three parallel lists (yes/no, MCQ, open-ended)
    such that the same index in each list corresponds to the same original question.
    """

    CATEGORIES = ['direct_attributes', 'GPT4V-hard', 'OCR', 'relative_position']

    def __init__(
        self,
        framing_json_path: str,
        image_root: str,
        seed: int = 42
    ):
        """
        Args:
            framing_json_path: Path to vstar_framing.json produced by reframe_vstar_questions.py
            image_root: Root directory containing category subdirs with images
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.framing_json_path = framing_json_path
        self.image_root = image_root
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load vstar_framing.json into a flat list ordered by key."""
        print(f"Loading V-Star framing data from: {self.framing_json_path}")
        with open(self.framing_json_path, 'r') as f:
            raw = json.load(f)

        items = []
        for key, record in raw.items():
            items.append({
                'key': key,
                'category': record['category'],
                'question_id': record['question_id'],
                'image_filename': record['image_filename'],
                'target_object': record.get('target_object', []),
                'bbox': record.get('bbox', []),
                'original': record['original'],     # {question, options}
                'reframed': record['reframed'],     # {open_ended, yes_no}
            })

        print(f"Loaded {len(items)} V-Star framing entries")
        return items

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _build_base(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Return the common fields shared by all three task types."""
        bbox = None
        all_bboxes = None
        if item['bbox']:
            bbox = item['bbox'][0]
            all_bboxes = item['bbox']

        return {
            'dataset': 'V-Star',
            'question_id': f"{item['category']}/{item['question_id']}",
            'image': os.path.join(self.image_root, item['category'], item['image_filename']),
            'category': item['category'],
            'target_object': item['target_object'],
            'bbox': bbox,
            'all_bboxes': all_bboxes,
        }

    def _format_mcq(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format item as a shuffled MCQ sample."""
        options = item['original']['options']
        correct_answer = options[0]  # first option is always correct

        shuffled_options = options.copy()
        random.shuffle(shuffled_options)

        correct_idx = shuffled_options.index(correct_answer)
        correct_letter = chr(65 + correct_idx)  # 'A', 'B', ...

        question = item['original']['question']
        formatted_text = question + "\n"
        for idx, opt in enumerate(shuffled_options):
            formatted_text += f"{chr(65 + idx)}. {opt}\n"

        base = self._build_base(item)
        base.update({
            'task_type': 'mcq',
            'text': formatted_text + MCQ_POSTFIX,
            'text_nopostfix': formatted_text,
            'answer': correct_letter,
            'answer_text': correct_answer,
            'options': shuffled_options,
        })
        return base

    def _format_yes_no(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format item as a yes/no sample using the reframed yes_no question."""
        yn = item['reframed']['yes_no']
        question = yn['question']
        answer = yn['answer']  # 'yes' or 'no'

        base = self._build_base(item)
        base.update({
            'task_type': 'yes_no',
            'text': question + YESNO_POSTFIX,
            'text_nopostfix': question,
            'answer': answer,
            'answer_text': answer,
            'options': ['yes', 'no'],
        })
        return base

    def _format_open_ended(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format item as an open-ended sample using the reframed open_ended question."""
        oe = item['reframed']['open_ended']
        question = oe['question']
        answer = oe['answer']

        base = self._build_base(item)
        base.update({
            'task_type': 'open_ended',
            'text': question + OPEN_ENDED_POSTFIX,
            'text_nopostfix': question,
            'answer': answer,
            'answer_text': answer,
            'options': [],
        })
        return base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n_samples: int):
        """
        Randomly pick n_samples items and return three parallel lists:
            (yn_samples, mcq_samples, oe_samples)
        Same index across all three lists corresponds to the same original question.
        Pass n_samples=-1 to use all items.
        """
        if n_samples == -1:
            chosen = self.data
        else:
            chosen = random.sample(self.data, min(n_samples, len(self.data)))

        yn_samples, mcq_samples, oe_samples = [], [], []
        for item in chosen:
            yn_samples.append(self._format_yes_no(item))
            mcq_samples.append(self._format_mcq(item))
            oe_samples.append(self._format_open_ended(item))

        print(f"Sampled {len(chosen)} V-Star framing questions (3 framings each)")
        return yn_samples, mcq_samples, oe_samples

    def sample_by_category(self, n_per_category: int):
        """
        Sample n_per_category items from each of the 4 categories and return
        three parallel lists: (yn_samples, mcq_samples, oe_samples).
        """
        chosen = []
        for cat in self.CATEGORIES:
            cat_items = [it for it in self.data if it['category'] == cat]
            if not cat_items:
                print(f"Warning: No items found for category: {cat}")
                continue
            n_take = min(n_per_category, len(cat_items))
            chosen.extend(random.sample(cat_items, n_take))

        yn_samples, mcq_samples, oe_samples = [], [], []
        for item in chosen:
            yn_samples.append(self._format_yes_no(item))
            mcq_samples.append(self._format_mcq(item))
            oe_samples.append(self._format_open_ended(item))

        print(f"Sampled {len(chosen)} V-Star framing questions by category ({n_per_category} per category)")
        return yn_samples, mcq_samples, oe_samples

class MMESampler(BaseSampler):
    """
    Sampler for MME dataset (Yes/No questions).
    Loads questions from JSONL and answers from corresponding txt files.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        answer_root: str,
        seed: int = 42
    ):
        """
        Args:
            jsonl_path: Path to MME JSONL file
            image_root: Root directory for MME images
            answer_root: Root directory for MME answer txt files (e.g., MME_Benchmark_release_version)
            seed: Random seed
        """
        super().__init__(seed)
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.answer_root = answer_root
        self.data = self._load_data()
    
    def _load_answers_from_txt(self, txt_path: str) -> Dict[str, str]:
        """
        Load answers from a txt file.
        Each line format: "Question text Yes/No"
        
        Returns:
            Dict mapping question text to answer (Yes/No)
        """
        answers = {}
        if not os.path.exists(txt_path):
            # Instead of using the provided txt_path, construct the path using the parent directory and "questions_answers_YN"
            parent_dir = os.path.dirname(txt_path)
            filename = os.path.basename(txt_path)
            txt_path = os.path.join(parent_dir, "questions_answers_YN", filename)
        
        # Try multiple encodings to handle various file formats
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if content is None:
            print(f"Warning: Could not decode file {txt_path} with any known encoding")
            return answers
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Remove the postfix if present
            if " Please answer yes or no." in line:
                line = line.replace(" Please answer yes or no.", "")
            
            # Split from the right to get the answer (Yes/No)
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                question_text = parts[0].strip()
                answer = parts[1].strip()
                answers[question_text] = answer

        return answers
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load all MME data with answers from txt files."""
        print(f"Loading MME data from: {self.jsonl_path}")
        print(f"Loading answers from: {self.answer_root}")
        
        all_data = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                all_data.append(data)
        
        print(f"Loaded {len(all_data)} MME samples from JSONL")
        
        # Load answers from txt files
        loaded_answers = 0
        for sample in all_data:
            # Extract category and image filename
            image_path = sample['image']  # e.g., "code_reasoning/0020.png"
            category = sample['category']
            image_filename = os.path.basename(image_path).replace('.png', '.txt').replace('.jpg', '.txt')
            
            # Construct path to txt file
            txt_path = os.path.join(self.answer_root, category, image_filename)
            
            # Load answers from txt
            answers_dict = self._load_answers_from_txt(txt_path)
            
            # Match question text to get answer
            question_text = sample['text']
            if question_text in answers_dict:
                sample['answer'] = answers_dict[question_text]
                loaded_answers += 1
            else:
                # Try without exact match (in case of slight formatting differences)
                sample['answer'] = ''
                raise ValueError(f"Answer not found for question: {question_text} in file: {txt_path}")
        
        print(f"Successfully loaded answers for {loaded_answers}/{len(all_data)} samples")
        return all_data
    
    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from MME dataset.
        
        Returns:
            List of samples with 'task_type': 'yes_no'
        """
        if n_samples == -1:
            n_samples = len(self.data)
        sampled = random.sample(self.data, min(n_samples, len(self.data)))
        
        formatted_samples = []
        for sample in sampled:
            formatted_samples.append({
                'task_type': 'yes_no',
                'dataset': 'MME',
                'question_id': sample['question_id'],
                'image': os.path.join(self.image_root, sample['image']),
                'text': sample['text'] + YESNO_POSTFIX,
                'text_nopostfix': sample['text'],
                'answer': sample.get('answer', ''),
                'category': sample.get('category', 'unknown'),
                'options': ['yes', 'no']
            })

        print(f"Sampled {len(formatted_samples)} MME yes/no questions")
        return formatted_samples


class POPESampler(BaseSampler):
    """
    Sampler for POPE dataset (Yes/No questions).
    Loads data from three POPE JSONL files (random, popular, adversarial).
    """
    
    def __init__(
        self,
        pope_root: str,
        image_root: str,
        seed: int = 42
    ):
        """
        Args:
            pope_root: Directory containing coco_pope_{category}.json files
            image_root: Root directory for COCO images (val2014)
            seed: Random seed
        """
        super().__init__(seed)
        self.pope_root = pope_root
        self.image_root = image_root
        self.categories = ['random', 'popular', 'adversarial']
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load POPE data from all three categories."""
        all_data = []
        
        for category in self.categories:
            filename = f"coco_pope_{category}.json"
            file_path = os.path.join(self.pope_root, filename)
            print(f"Loading POPE {category} data from: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Add category info to distinguish source
                        data['pope_category'] = category
                        # Ensure question_id is unique by prefixing category
                        # (original ids overlap between files)
                        data['original_question_id'] = data.get('question_id')
                        data['question_id'] = f"{category}_{data.get('question_id')}"
                        all_data.append(data)
                    except json.JSONDecodeError:
                        continue
                        
        print(f"Loaded {len(all_data)} POPE samples total")
        return all_data
    
    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from POPE.
        
        Returns:
            List of samples with 'task_type': 'yes_no'
        """
        if n_samples == -1:
             sampled = self.data
        else:
             sampled = random.sample(self.data, min(n_samples, len(self.data)))
        
        formatted_samples = []
        for sample in sampled:
            formatted_samples.append({
                'task_type': 'yes_no',
                'dataset': 'POPE',
                'question_id': sample['question_id'],
                'image': os.path.join(self.image_root, sample['image']),
                'text': sample['text'] + YESNO_POSTFIX,
                'text_nopostfix': sample['text'],
                'answer': sample['label'],
                'category': sample['pope_category'],
                'options': ['yes', 'no']
            })
            
        print(f"Sampled {len(formatted_samples)} POPE yes/no questions")
        return formatted_samples

class HallusionBenchSampler(BaseSampler):
    """
    Sampler for HallusionBench dataset.
    Yes/No visual question answering benchmark with two categories:
    - VD (Visual Dependent): questions that require the image to answer
    - VS (Visual Supplement): questions where the image supplements text
    Subcategories: chart, figure, illusion, map, math, ocr, table, video
    """

    def __init__(
        self,
        hallusion_root: str,
        image_cache_dir: str = None,
        seed: int = 42
    ):
        """
        Args:
            hallusion_root: Directory containing image-00000-of-00001.parquet
            image_cache_dir: Directory to cache decoded images. If None, uses hallusion_root/images
            seed: Random seed
        """
        super().__init__(seed)
        self.hallusion_root = hallusion_root
        self.parquet_path = os.path.join(hallusion_root, 'image-00000-of-00001.parquet')
        self.image_cache_dir = image_cache_dir or os.path.join(hallusion_root, 'images')
        os.makedirs(self.image_cache_dir, exist_ok=True)
        self.categories = ['VD', 'VS']
        self.subcategories = ['chart', 'figure', 'illusion', 'map', 'math', 'ocr', 'table', 'video']
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load all HallusionBench data from the parquet file and decode images to disk."""
        import pyarrow.parquet as pq

        print(f"Loading HallusionBench data from: {self.parquet_path}")

        table = pq.read_table(self.parquet_path)
        all_data = []

        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            if row['visual_input'] == 0:
                continue  # skip non-visual questions
            # Build a unique ID from category/subcategory/set_id/figure_id/question_id
            qid = f"{row['category']}_{row['subcategory']}_{row['set_id']}_{row['figure_id']}_{row['question_id']}"

            # Decode and cache image
            # Use the filename structure to organize: e.g. VS/chart/0_1.png
            rel_path = row['filename'].lstrip('./')  # e.g. "VS/chart/0_1.png"
            image_path = os.path.join(self.image_cache_dir, rel_path)

            if not os.path.exists(image_path):
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image_bytes = row['image'].get('bytes') if isinstance(row['image'], dict) else None
                if image_bytes:
                    with open(image_path, 'wb') as img_f:
                        img_f.write(image_bytes)
                else:
                    print(f"Warning: No image bytes for row {i} ({qid})")
                    continue

            # gt_answer: '0' = No, '1' = Yes
            gt_answer = 'Yes' if row['gt_answer'] == '1' else 'No'

            sample = {
                'question_id': qid,
                'category': row['category'],
                'subcategory': row['subcategory'],
                'visual_input': row['visual_input'],
                'set_id': row['set_id'],
                'figure_id': row['figure_id'],
                'question': row['question'],
                'gt_answer': gt_answer,
                'gt_answer_details': row.get('gt_answer_details', ''),
                'sample_note': row.get('sample_note', ''),
                'image_path': image_path,
            }
            assert gt_answer in ['Yes', 'No'], f"Unexpected gt_answer: {gt_answer} in sample {qid}"

            all_data.append(sample)

        print(f"Loaded {len(all_data)} HallusionBench samples total")
        for cat in self.categories:
            count = sum(1 for s in all_data if s['category'] == cat)
            print(f"  {cat}: {count}")
        for subcat in self.subcategories:
            count = sum(1 for s in all_data if s['subcategory'] == subcat)
            if count > 0:
                print(f"    {subcat}: {count}")

        return all_data

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from HallusionBench dataset.

        Returns:
            List of samples with same keys as VstarSampler output:
            task_type, dataset, question_id, image, text, text_nopostfix,
            answer, answer_text, category, target_object, bbox, all_bboxes
        """
        if n_samples == -1:
            n_samples = len(self.data)

        sampled = random.sample(self.data, min(n_samples, len(self.data)))

        formatted_samples = []
        for sample in sampled:
            formatted_text = sample['question']
            
            formatted_text_wpostfix = formatted_text + YESNO_POSTFIX

            formatted_sample = {
                'task_type': 'yesno',
                'dataset': 'HallusionBench',
                'question_id': sample['question_id'],
                'image': sample['image_path'],
                'text': formatted_text_wpostfix,
                'text_nopostfix': formatted_text,
                'answer': sample['gt_answer'],
                'answer_text': sample['gt_answer'],
                'category': sample['category'],
                'target_object': [],
                'bbox': None,
                'all_bboxes': None,
                'options': ['yes', 'no'],
            }
            formatted_samples.append(formatted_sample)

        print(f"Sampled {len(formatted_samples)} HallusionBench questions")
        return formatted_samples

    def sample_by_category(
        self,
        n_samples_per_category: int
    ) -> List[Dict[str, Any]]:
        """
        Sample n_samples_per_category from each category (VD, VS).

        Args:
            n_samples_per_category: Number of samples per category

        Returns:
            Combined list of balanced samples across categories
        """
        all_samples = []

        for category in self.categories:
            category_samples = [s for s in self.data if s['category'] == category]

            if not category_samples:
                print(f"Warning: No samples found for category: {category}")
                continue

            n_take = min(n_samples_per_category, len(category_samples))
            sampled = random.sample(category_samples, n_take)

            for sample in sampled:
                formatted_text = sample['question']
                formatted_text_wpostfix = formatted_text + YESNO_POSTFIX

                formatted_sample = {
                    'task_type': 'yesno',
                    'dataset': 'HallusionBench',
                    'question_id': sample['question_id'],
                    'image': sample['image_path'],
                    'text': formatted_text_wpostfix,
                    'text_nopostfix': formatted_text,
                    'answer': sample['gt_answer'],
                    'answer_text': sample['gt_answer'],
                    'category': sample['category'],
                    'target_object': [],
                    'bbox': None,
                    'all_bboxes': None,
                    'options': ['yes', 'no'],
                }
                all_samples.append(formatted_sample)

        print(f"Sampled {len(all_samples)} HallusionBench questions ({n_samples_per_category} per category)")
        return all_samples

    def sample_by_subcategory(
        self,
        n_samples_per_subcategory: int
    ) -> List[Dict[str, Any]]:
        """
        Sample n_samples_per_subcategory from each subcategory.

        Args:
            n_samples_per_subcategory: Number of samples per subcategory

        Returns:
            Combined list of balanced samples across subcategories
        """
        all_samples = []

        for subcategory in self.subcategories:
            subcat_samples = [s for s in self.data if s['subcategory'] == subcategory]

            if not subcat_samples:
                continue

            n_take = min(n_samples_per_subcategory, len(subcat_samples))
            sampled = random.sample(subcat_samples, n_take)

            for sample in sampled:
                formatted_text = sample['question']
                formatted_text_wpostfix = formatted_text + YESNO_POSTFIX

                formatted_sample = {
                    'task_type': 'yesno',
                    'dataset': 'HallusionBench',
                    'question_id': sample['question_id'],
                    'image': sample['image_path'],
                    'text': formatted_text_wpostfix,
                    'text_nopostfix': formatted_text,
                    'answer': sample['gt_answer'],
                    'answer_text': sample['gt_answer'],
                    'category': sample['category'],
                    'target_object': [],
                    'bbox': None,
                    'all_bboxes': None,
                    'options': ['yes', 'no'],
                }
                all_samples.append(formatted_sample)

        print(f"Sampled {len(all_samples)} HallusionBench questions ({n_samples_per_subcategory} per subcategory)")
        return all_samples



class RealWorldQASampler(BaseSampler):
    """Sampler for RealWorldQA dataset (MCQ, parquet with embedded images)."""

    def __init__(self, realworldqa_root: str, seed: int = 42):
        super().__init__(seed)
        self.realworldqa_root = realworldqa_root
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        import glob
        import pandas as pd

        parquet_files = sorted(glob.glob(os.path.join(self.realworldqa_root, 'data', '*.parquet')))
        print(f"Loading RealWorldQA from {len(parquet_files)} parquet files...")

        images_dir = os.path.join(self.realworldqa_root, 'images')
        os.makedirs(images_dir, exist_ok=True)

        all_data = []
        count_MCQ = 0
        count_MCQ_tt = 0
        global_idx = 0
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                img_path = os.path.join(images_dir, f'{global_idx}.webp')
                if not os.path.exists(img_path):
                    with open(img_path, 'wb') as f:
                        f.write(row['image']['bytes'])

                if 'Please answer directly with only the letter of the correct option and nothing else.' in row['question']:
                    row['question'] = row['question'].replace('Please answer directly with only the letter of the correct option and nothing else.', '').strip()
                    all_data.append({
                        'idx': global_idx,
                        'question': row['question'],
                        'answer': row['answer'],
                        'image': img_path,
                        'task_type': 'mcq',
                    })

                global_idx += 1

                if row['answer'] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                    count_MCQ += 1
                if 'Please answer directly with only the letter of the correct option and nothing else.' in row['question']:
                    count_MCQ_tt += 1

        print(f"Loaded {len(all_data)} RealWorldQA samples ({count_MCQ} MCQ, {count_MCQ_tt} MCQ_tt)")
        return all_data

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        if n_samples == -1:
            n_samples = len(self.data)
        sampled = random.sample(self.data, min(n_samples, len(self.data)))

        formatted = []
        for s in sampled:
            question_text = s['question']
            formatted.append({
                'task_type': 'mcq',
                'dataset': 'RealWorldQA',
                'question_id': f"realworldqa_{s['idx']}",
                'image': s['image'],
                'text': question_text + MCQ_POSTFIX,
                'text_nopostfix': question_text,
                'answer': s['answer'],
                'category': 'realworldqa',
                'options': [],
            })

        print(f"Sampled {len(formatted)} RealWorldQA questions")
        return formatted


class MMMUProSampler(BaseSampler):
    """Sampler for MMMU-Pro standard 4-option set (MCQ, single-image, parquet)."""

    def __init__(self, mmmupro_root: str, seed: int = 42):
        super().__init__(seed)
        self.mmmupro_root = mmmupro_root
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        import ast
        import glob
        import pandas as pd

        data_dir = os.path.join(self.mmmupro_root, 'vision')
        parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
        print(f"Loading MMMU-Pro from {len(parquet_files)} parquet files...")

        images_dir = os.path.join(self.mmmupro_root, 'images')
        os.makedirs(images_dir, exist_ok=True)

        all_data = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                # if row.get('image_1') is None or row.get('image_2') is not None:
                #     continue

                sample_id = row['id']
                img_path = os.path.join(images_dir, f'{sample_id}.png')
                if not os.path.exists(img_path):
                    with open(img_path, 'wb') as f:
                        f.write(row['image']['bytes'])

                options = ast.literal_eval(row['options'])

                all_data.append({
                    'id': sample_id,
                    'question': "", # question is already in the image as screen text, so we leave it empty here
                    'options': options,
                    'answer': row['answer'],
                    'image': img_path,
                    'subject': row.get('subject', ''),
                })

        print(f"Loaded {len(all_data)} MMMU-Pro samples (single-image)")
        return all_data

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        if n_samples == -1:
            n_samples = len(self.data)
        sampled = random.sample(self.data, min(n_samples, len(self.data)))

        formatted = []
        for s in sampled:
            options = s['options']
            option_str = '\n'.join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            question_text = f"{s['question']}\n{option_str}"

            formatted.append({
                'task_type': 'mcq',
                'dataset': 'MMMU-Pro',
                'question_id': s['id'],
                'image': s['image'],
                'text': question_text + MCQ_POSTFIX,
                'text_nopostfix': question_text,
                'answer': s['answer'],
                'category': s['subject'],
                'options': options,
            })

        print(f"Sampled {len(formatted)} MMMU-Pro questions")
        return formatted


class HRBenchSampler(BaseSampler):
    """
    Sampler for HR-Bench (High-Resolution Benchmark) dataset.
    Supports 4k and 8k resolution variants.
    Each sample is an MCQ with options A/B/C/D, and images are base64-encoded in the TSV.
    """

    def __init__(
        self,
        hrbench_root: str,
        resolution: str = '4k',
        image_cache_dir: str = None,
        seed: int = 42
    ):
        """
        Args:
            hrbench_root: Root directory containing hr_bench_4k.tsv and hr_bench_8k.tsv
            resolution: '4k' or '8k'
            image_cache_dir: Directory to cache decoded images. If None, uses hrbench_root/images_{resolution}
            seed: Random seed
        """
        super().__init__(seed)
        assert resolution in ('4k', '8k'), f"resolution must be '4k' or '8k', got '{resolution}'"
        self.hrbench_root = hrbench_root
        self.resolution = resolution
        self.tsv_path = os.path.join(hrbench_root, f'hr_bench_{resolution}.tsv')
        self.image_cache_dir = image_cache_dir or os.path.join(hrbench_root, f'images_{resolution}')
        os.makedirs(self.image_cache_dir, exist_ok=True)
        self.categories = ['single', 'cross']
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load all HR-Bench data from the TSV file and decode images to disk."""
        print(f"Loading HR-Bench {self.resolution} data from: {self.tsv_path}")

        csv.field_size_limit(sys.maxsize)

        all_data = []
        with open(self.tsv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t', quotechar='"')
            for row in reader:
                idx = row['index']
                image_path = os.path.join(self.image_cache_dir, f'{idx}.jpg')

                # Decode and cache image if not already on disk
                if not os.path.exists(image_path):
                    image_bytes = base64.b64decode(row['image'])
                    with open(image_path, 'wb') as img_f:
                        img_f.write(image_bytes)

                sample = {
                    'question_id': idx,
                    'category': row['category'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'options': [row['A'], row['B'], row['C'], row['D']],
                    'answer_text': row[row['answer']],
                    'cycle_category': row.get('cycle_category', ''),
                    'image_path': image_path,
                }
                all_data.append(sample)

        print(f"Loaded {len(all_data)} HR-Bench {self.resolution} samples total")
        for category in self.categories:
            count = sum(1 for s in all_data if s['category'] == category)
            print(f"  {category}: {count}")

        return all_data

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample n_samples from HR-Bench dataset.

        Returns:
            List of samples with same keys as VstarSampler output:
            task_type, dataset, question_id, image, text, text_nopostfix,
            answer, answer_text, category, target_object, bbox, all_bboxes
        """
        if n_samples == -1:
            n_samples = len(self.data)

        sampled = random.sample(self.data, min(n_samples, len(self.data)))

        formatted_samples = []
        for sample in sampled:
            options = sample['options']
            answer_letter = sample['answer']  # original correct letter (A/B/C/D)
            correct_answer = sample['answer_text']

            # Shuffle options and track correct answer position
            shuffled_options = options.copy()
            random.shuffle(shuffled_options)
            correct_idx = shuffled_options.index(correct_answer)
            correct_letter = chr(65 + correct_idx)

            # Format question with MCQ options
            formatted_text = sample['question'] + "\n"
            for idx, option in enumerate(shuffled_options):
                letter = chr(65 + idx)
                formatted_text += f"{letter}. {option}\n"
            formatted_text_wpostfix = formatted_text + MCQ_POSTFIX

            formatted_sample = {
                'task_type': 'mcq',
                'dataset': f'HR-Bench-{self.resolution}',
                'question_id': sample['question_id'],
                'image': sample['image_path'],
                'text': formatted_text_wpostfix,
                'question': sample['question'],
                'text_nopostfix': formatted_text,
                'answer': correct_letter,
                'answer_text': correct_answer,
                'category': sample['category'],
                'target_object': [],
                'bbox': None,
                'all_bboxes': None,
                'options': shuffled_options,
            }
            formatted_samples.append(formatted_sample)

        print(f"Sampled {len(formatted_samples)} HR-Bench {self.resolution} MCQ questions")
        return formatted_samples

    def sample_by_category(
        self,
        n_samples_per_category: int
    ) -> List[Dict[str, Any]]:
        """
        Sample n_samples_per_category from each category (single, cross).

        Args:
            n_samples_per_category: Number of samples per category

        Returns:
            Combined list of balanced samples across categories
        """
        all_samples = []

        for category in self.categories:
            category_samples = [s for s in self.data if s['category'] == category]

            if not category_samples:
                print(f"Warning: No samples found for category: {category}")
                continue

            n_take = min(n_samples_per_category, len(category_samples))
            sampled = random.sample(category_samples, n_take)

            for sample in sampled:
                options = sample['options']
                correct_answer = sample['answer_text']

                shuffled_options = options.copy()
                random.shuffle(shuffled_options)
                correct_idx = shuffled_options.index(correct_answer)
                correct_letter = chr(65 + correct_idx)

                formatted_text = sample['question'] + "\n"
                for idx, option in enumerate(shuffled_options):
                    letter = chr(65 + idx)
                    formatted_text += f"{letter}. {option}\n"
                formatted_text += MCQ_POSTFIX

                formatted_sample = {
                    'task_type': 'mcq',
                    'dataset': f'HR-Bench-{self.resolution}',
                    'question_id': sample['question_id'],
                    'image': sample['image_path'],
                    'text': formatted_text,
                    'text_nopostfix': sample['question'],
                    'answer': correct_letter,
                    'answer_text': correct_answer,
                    'category': sample['category'],
                    'target_object': [],
                    'bbox': None,
                    'all_bboxes': None,
                    'options': shuffled_options,
                }
                all_samples.append(formatted_sample)

        print(f"Sampled {len(all_samples)} HR-Bench {self.resolution} MCQ questions ({n_samples_per_category} per category)")
        return all_samples
