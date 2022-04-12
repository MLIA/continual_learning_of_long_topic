import logging
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from torch import optim

from lire.function_tools import gradient_tools




tokenizer = T5Tokenizer.from_pretrained("t5-small")


positive_document = tokenizer("true", return_tensors="pt").input_ids.cuda()
negative_document = tokenizer("false", return_tensors="pt").input_ids.cuda()

print("The id of positive document is ", positive_document)
print("The id of negative document is ", negative_document)

print("Example of one backward for a query and a document : \n")

queries = "What is the name of the 2019 epidemic ?"
document = "In 2019 the coronavirus lead to the largest pandemic of the 21th century."

print("\t The query is ", queries)
print("\t The document is ", document)
tokenized_sentence = tokenizer(queries+"</s>"+document, return_tensors="pt").input_ids.cuda()
 
print("Tokenized concatenation of query document ", tokenized_sentence)

detokenized_sentence = tokenizer.convert_ids_to_tokens(tokenized_sentence.squeeze().tolist())
print("\tDetokenized sentence: ", detokenized_sentence)

print("\n Example using T5: \n")

model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()

adam_optimizer = optim.Adam(model.parameters())

outputs = model(input_ids=tokenized_sentence, labels=positive_document)
print("Loss ", outputs.loss)
print("score on true token before training ", outputs.logits[0,1,positive_document[0][0]].item())
print("score on true token before training ", outputs.logits[0,0,positive_document[0][0]].item())
print("max item ", outputs.logits[0,0,:].argmax())
outputs = model.generate(input_ids=tokenized_sentence)
print(tokenizer.convert_ids_to_tokens(outputs.squeeze().tolist()))
for i in range(100):

    adam_optimizer.zero_grad()

    outputs = model(input_ids=tokenized_sentence, labels=positive_document)
    outputs.loss.backward()
    print(i, outputs.loss.item())
    adam_optimizer.step()


outputs = model(input_ids=tokenized_sentence, labels=positive_document)
print("Loss ", outputs.loss)
print("score on true token after training ", outputs.logits[0,1,positive_document[0][0]].item())
print("score on true token after training ", outputs.logits[0,0,positive_document[0][0]].item())
print("max item ", outputs.logits[0,0,:].argmax())
print("Generating sentence ")
outputs = model.generate(input_ids=tokenized_sentence)
print(tokenizer.convert_ids_to_tokens(outputs.squeeze().tolist()))

class local_model(object):
    def __init__(self, hf_model):
        self.hf_model = hf_model
    
    def __call__(self, x, y):
        outputs = self.hf_model(input_ids=x, labels=y)
        return outputs.logits[0, 0, positive_document[0][0]]
    
    def parameters(self):
        return self.hf_model.parameters()

lm = local_model(model)
y = lm(tokenized_sentence, positive_document)
print(y)

fisher_diag = gradient_tools.estimate_f
isher_diag([tokenized_sentence], [positive_document], lm)
print([layer.shape for layer in fisher_diag])


