from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Step 1：实例化一个空白的BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Step 2：实例化一个BPE tokenizer 的训练器 trainer 这里 special_tokens 的顺序决定了其id排序
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=1,
    show_progress=True,
    vocab_size=50000,
)

# Step 3：定义预分词规则（比如以空格预切分）
tokenizer.pre_tokenizer = Whitespace()

# Step 4：加载数据集 训练tokenizer
files = ["dataset/test.jsonl", "dataset/train.jsonl"]
tokenizer.train(files, trainer)

# Step 5：保存 tokenizer
tokenizer.save("tokenizer/tokenizer.json")
