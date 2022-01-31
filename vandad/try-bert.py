import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer('bert-base-uncased', cache_dir='_cache')


tokenizer = BertTokenizer('_cache/vocabs/bert-base-multilingual-cased-vocab.txt', do_lower_case=False)


# Tokenized input
text = "[CLS] Who was Jim Henson ? #lol [SEP] Jim Henson was a puppeteer [SEP]"
text = "Shoutout to my mom for being hella ☻ supportive of me"
text = "@hamed Shoutout to my mom for being hella ☻ supportive of me"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 11
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '#', 'lo', '##l', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
attention_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
attention_tensors = torch.tensor([attention_ids])






# Load pre-trained model (weights)
model = BertModel.from_pretrained('_cache/models/bert-base-cased.tar.gz', cache_dir='_cache')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
attention_tensors = attention_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors, attention_tensors)
    
data_out = encoded_layers[int(10)].detach().cpu().numpy()
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12


predictions = model(tokens_tensor, segments_tensors)
predictions.backward(torch.zeros(1, 14, 30522, dtype=torch.float).to('cuda'))









# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'