import tiktoken
import torch
def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # print(encoded,", Shape: ", len(encoded))
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print(encoded_tensor,", Shape: ", len(encoded_tensor))
    return encoded_tensor

def token_to_text(tsr, tokenizer):
    flat = tsr.squeeze(0)
    text= tokenizer.decode(flat.tolist())
    return text

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # get only last step
        logits = logits[:,-1,:]
        idx_next = torch.argmax(logits,dim=-1,keepdim=True) #(batch,1)
        idx=torch.cat((idx,idx_next),dim=1)
    return idx
