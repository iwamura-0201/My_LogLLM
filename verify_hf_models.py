import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel

def check_model(model_id, model_class=None):
    print(f"Checking {model_id}...")
    try:
        AutoTokenizer.from_pretrained(model_id)
        if model_class:
            # Load with low_cpu_mem_usage to avoid OOM during check
            model_class.from_pretrained(model_id, low_cpu_mem_usage=True)
        print(f"SUCCESS: {model_id} is accessible.")
        return True
    except Exception as e:
        print(f"FAILED: Could not load {model_id}.")
        print(f"Error: {e}")
        return False

def main():
    print("--- Verifying Hugging Face Models ---")
    
    # Check BERT
    bert_ok = check_model("bert-base-uncased", BertModel)
    
    # Check Llama
    # Llama 3 is gated, so this verifies token presence/access
    llama_ok = check_model("meta-llama/Meta-Llama-3-8B", AutoModelForCausalLM)

    if not llama_ok:
        print("\n[!] Llama 3 access failed.")
        print("Please ensure you have:")
        print("1. Accepted the license at https://huggingface.co/meta-llama/Meta-Llama-3-8B")
        print("2. Logged in via terminal: `huggingface-cli login`")
    
    if bert_ok and llama_ok:
        print("\nAll models verified successfully!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
