from example import setup_model_parallel, load, LLaMA
import torch

def main():
    ckpt_dir = "/shared/katop1234/LLM/llama/7B/"
    tokenizer_path = "/shared/katop1234/LLM/llama/tokenizer.model"
    max_seq_len = 512
    max_batch_size = 32

    # Set up model parallelism
    local_rank, world_size = setup_model_parallel()

    # Load the model
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    # Define your specific prompt
    specific_prompt = "A comprehensive survey of the current state of the art in neural networks."

    # Create a list containing only your specific prompt
    prompts = [specific_prompt]

    # Set generation parameters if needed
    temperature = 0.8
    top_p = 0.95

    # Generate results for your specific prompt
    results, activations = generator.generate(
        prompts, max_gen_len=1024, temperature=temperature, top_p=top_p
    )
    
    linearized_activations = {}
    for token_index, token_activations in activations.items():
        flattened_tensors = [activation.flatten() for activation in token_activations]
        linearized_vector = torch.cat(flattened_tensors)
        print("For token index", token_index, "the linearized vector has shape", linearized_vector.shape)
        linearized_activations[token_index] = linearized_vector
    
    import torch.nn.functional as F

    # Get the first linearized activation
    first_index = sorted(linearized_activations.keys())[0] + 1 # +1 because the first token is the prompt and has larger shape
    first_activation = linearized_activations[first_index]
    
    tokenizer = generator.tokenizer
    generated_text = results[0]
    token_ids = tokenizer.encode(generated_text, bos=False, eos=False)
    similarities = []
    for idx, activation in linearized_activations.items():
        if idx == first_index - 1:
            continue
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(first_activation.unsqueeze(0), activation.unsqueeze(0), dim=1)
        similarities.append(similarity)
        token_id = token_ids[idx] if idx < len(token_ids) else None
        token_str = tokenizer.decode([token_id]) if token_id is not None else "<UNKNOWN>"
        print(f"Cosine similarity between first activation and activation {idx} (token: {token_str}): {similarity.item()}")

    # Print the results
    for result in results:
        print(result)
        print("\n==================================\n")
    
    import matplotlib.pyplot as plt
    
    plt.plot([i.cpu().numpy() for i in similarities])
    plt.xlabel('Token Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarities with the First Activation')
    plt.grid(True)

    # Save the figure
    plt.savefig('cosine_similarities.png')

if __name__ == "__main__":
    main()
