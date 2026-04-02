WORK_DIR=$(pwd)
DATA_DIR="$WORK_DIR/data"

# GQA
mkdir -p $DATA_DIR/gqa
cd $DATA_DIR/gqa
# wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip 
# unzip images.zip
# rm images.zip

wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip
rm sceneGraphs.zip

wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip questions1.2.zip
rm questions1.2.zip
mkdir data 
mv val_balanced_questions.json data/val_pos_multiobj_balanced_questions_full.json

# HallusionBench
mkdir -p $DATA_DIR/HallusionBench
cd $DATA_DIR/HallusionBench
wget https://huggingface.co/datasets/lmms-lab/HallusionBench/resolve/main/data/image-00000-of-00001.parquet

# HRBench
mkdir -p $DATA_DIR/hrbench
cd $DATA_DIR/hrbench
wget https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.parquet
wget https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv

# MME
mkdir -p $DATA_DIR/MME
cd $DATA_DIR/MME
wget https://huggingface.co/datasets/darkyarding/MME/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip
rm MME_Benchmark_release_version.zip
mv MME_Benchmark_release_version/MME_Benchmark/* MME_Benchmark_release_version/

# MMMU Pro
hf download MMMU/MMMU_Pro --local-dir $DATA_DIR/mmmu_pro --repo-type dataset

# POPE
mkdir -p $DATA_DIR/pope
cd $DATA_DIR/pope
wget http://images.cocodataset.org/zips/val2014.zip 
unzip val2014.zip
rm val2014.zip
mkdir coco
cd coco 
wget https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json

# realworld-qa
hf download xai-org/RealworldQA --local-dir $DATA_DIR/realworldqa --repo-type dataset

# Seed-Bench
mkdir -p $DATA_DIR/seedbench
cd $DATA_DIR/seedbench
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip
unzip SEED-Bench-image.zip
rm SEED-Bench-image.zip
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench.json

# vstar
hf download craigwu/vstar_bench --local-dir $DATA_DIR/vstar_bench --repo-type dataset