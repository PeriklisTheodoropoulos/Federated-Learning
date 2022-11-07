import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.n_hidden = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)


        #self.hidden=(torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
    
    def forward(self, features, captions):
        captions=self.embed(captions[:,:-1])
        
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        outputs, _ = self.lstm(inputs)
        
        outputs = self.fc(outputs)
        
        return outputs
        
    def sample(self, inputs, states=None, max_len=20):
        predicted_list=[]
        for i in range(max_len):
            output_lstm,states=self.lstm(inputs,states)
            outputs = self.fc(output_lstm.squeeze(1))  
            
                
            target=outputs.max(1)[1]
            
            predicted_list.append(target.item())
            # We predicted the <end> word, so there is no further prediction to do
            if (target == 1):
                break
            inputs=self.embed(target).unsqueeze(1)
        return predicted_list
            
            
            
            
            