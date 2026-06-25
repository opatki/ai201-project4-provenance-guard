# run from the project root with the .venv active
from dotenv import load_dotenv; load_dotenv()
from signals.perplexity import compute_perplexity_score

print(compute_perplexity_score("The quick brown fox jumps over the lazy dog."))
print(compute_perplexity_score("Furthermore, it is important to note that this multifaceted approach underscores the testament to meticulous planning."))
