# imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import cv2
import csv
from pathlib import Path


# load in the pages
class ManuscriptDataset(Dataset):
    
    def __init__(self, imgPaths: List[str], labels: List[int], transform=None):

        self.imgPaths = imgPaths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):

        return len(self.imgPaths)

    def __getitem__(self, idx):

        img_path = self.imgPaths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)

        # img does not exist
        if img is None:
            raise RuntimeError("failed to load img")
        

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        
        return img, label
    

# using a convolutional neural network
# great for image processing
class ScribeIdentifyCNN(nn.Module):

    def __init__(self, numScribes: int, weights: bool = True):
        super(ScribeIdentifyCNN, self).__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights else None)

        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        numFeatures = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(numFeatures, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, numScribes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    

# look at the different letters; image processing step
class HandwritingFeatureExtractor:

    @staticmethod
    def preprocess_manuscript(image_path: str) -> np.ndarray:
       
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # remove noise
        img = cv2.fastNlMeansDenoising(img, h=10)
        
        # enhance contrast (makes it easier to use)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # binarization (otsu)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return img
    
    @staticmethod
    def extractStrokeFeatures(image: np.ndarray) -> Dict[str, float]:

        # skeletonize for stroke centerlines
        skeleton = cv2.ximgproc.thinning(image)
        
        # calc stroke width distribution
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        strokeWdths = dist_transform[skeleton > 0]
        
        features = {
            'meanStrokeWidth': np.mean(strokeWdths) if len(strokeWdths) > 0 else 0,
            'stdStrokeWidth': np.std(strokeWdths) if len(strokeWdths) > 0 else 0,
            'strokeDensity': np.sum(skeleton > 0) / skeleton.size
        }
        
        return features


class scribeIdentifier:
    
    def __init__(self, numScribes: int, device: str = None):
        self.numScribes = numScribes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ScribeIdentifyCNN(numScribes).to(self.device)
        
        # Data augmentation and normalization
        self.train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.valTransform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.history = {'trainLoss': [], 'valLoss': [], 'valAcc': []}
    
    def prepareData(self, dataDir: str, test_size: float = 0.2):

        imgPaths = []
        labels = []
        scribeNames = []
        
        data_path = Path(dataDir)
        for scribeIdx, scribeDir in enumerate(sorted(data_path.iterdir())):
            if scribeDir.is_dir():
                scribeNames.append(scribeDir.name)
                for img_path in list(scribeDir.glob('*.JPG')) + list(scribeDir.glob('*.png')):
                    imgPaths.append(str(img_path))
                    labels.append(scribeIdx)
        
        # split data
        xTrain, xVal, yTrain, yVal = train_test_split(
            imgPaths, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # training and validation datasets
        trainDataset = ManuscriptDataset(xTrain, yTrain, self.train_transform)
        valDataset = ManuscriptDataset(xVal, yVal, self.valTransform)
        
        # dataloaders
        trainLoader = DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=2)
        valLoader = DataLoader(valDataset, batch_size=8, shuffle=False, num_workers=2)
        
        return trainLoader, valLoader, scribeNames
    
    def train(self, trainLoader, valLoader, epochs: int = 20, lr: float = 0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        bestValAcc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            trainLoss = 0.0
            
            for images, labels in trainLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                trainLoss += loss.item()
            
            avgTrainLoss = trainLoss / len(trainLoader)
            
            # Validation phase
            valLoss, valAcc = self.evaluate(valLoader, criterion)
            
            # Update learning rate
            scheduler.step(valLoss)
            
            # Save history
            self.history['trainLoss'].append(avgTrainLoss)
            self.history['valLoss'].append(valLoss)
            self.history['valAcc'].append(valAcc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avgTrainLoss:.4f}')
            print(f'  Val Loss: {valLoss:.4f}, Val Acc: {valAcc:.4f}')
            
            # save best model
            if valAcc > bestValAcc:
                bestValAcc = valAcc
                torch.save(self.model.state_dict(), 'best_scribe_model.pth')
                print(f'  New best model saved!')
    
    def evaluate(self, dataLoader, criterion):
        """Evaluate model on validation/test set"""
        self.model.eval()
        totalLoss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                totalLoss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avgLoss = totalLoss / len(dataLoader)
        accuracy = correct / total
        
        return avgLoss, accuracy
    

    def predict(self, imgPath: str, scribeNames: List[str]) -> Tuple[str, Dict[str, float]]:
        """Predict copyist for a single manuscript page"""
        self.model.eval()
        
        # Load and transform image
        image = Image.open(imgPath).convert('RGB')
        imgTensor = self.valTransform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(imgTensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
        
        # Get prediction
        preditIdx = torch.argmax(probabilities).item()
        predictedScribe = scribeNames[preditIdx]
        
        # Create probability dictionary
        prob_dict = {
            scribeNames[i]: probabilities[i].item() 
            for i in range(len(scribeNames))
        }
        
        return predictedScribe, prob_dict
    

    def plot_training_history(self):
        """Visualize training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['trainLoss'], label='Train Loss')
        ax1.plot(self.history['valLoss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['valAcc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def predictFolder(self, folderPath: str, scribeNames: List[str], outputCSV: str = "predictions.csv"):

        folder = Path(folderPath)
        imgPaths = sorted([p for p in folder.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        if len(imgPaths) == 0:
            print("no value imgs found in folder")
            return
    
        results = []

        for imgPath in imgPaths:
            try:
                predicted, prob_dict = self.predict(str(imgPath), scribeNames)

                # record full row (one row per file)
                row = {
                    "filename": imgPath.name,
                    "predicted_scribe": predicted
                }

                # add all probabilities
                for scribe in scribeNames:
                    row[f"prob_{scribe}"] = prob_dict[scribe]

                    results.append(row)

                    print(f"{imgPath.name} -> {predicted}")

            except Exception as e:
                print(f"Could not process {imgPath.name}: {e}")

            # Save CSV
            with open(outputCSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["filename", "predicted_scribe"] +
                            [f"prob_{s}" for s in scribeNames]
                )
                writer.writeheader()
                writer.writerows(results)

            print(f"\nSaved predictions to {outputCSV}")


# ------------------------------------
# -------- actually run it now -------
# ------------------------------------
if __name__ == "__main__":

    numScribes = 5 # ADJUST TO MATCH THE NUMBER OF SCRIBES DONT FORGET
                   # five diff. manuscripts to span the middle ages.
    identifier = scribeIdentifier(numScribes=numScribes)
    
    dataDirectory = "manuscript_data"
    trainLoader, valLoader, scribeNames = identifier.prepareData(dataDirectory)
    
    print(f"found {len(scribeNames)} scribes: {scribeNames}")
    print(f"training samples: {len(trainLoader.dataset)}")
    print(f"validtion samples: {len(valLoader.dataset)}")
    
    # model training
    identifier.train(trainLoader, valLoader, epochs=20, lr=0.0001)
    
    # training history
    identifier.plot_training_history()
    
    testFolder = "test_scribes/mixbag_test"

    identifier.predictFolder(
        folderPath=testFolder,
        scribeNames=scribeNames,
        outputCSV="test_predictions.csv"
    )


    # predict one new page. old code.
    # CHANGE FOR EACH IMAGE CANT FIND A BETTER WAY TO DO THIS
    #testImg = "C:/Users/sophi/Desktop/TECH/senior/fall 2025/machine learning/Manuscript-Analysis-Capstone/test_scribes/test_scribe_I/lfg_f257v.JPG"
    #predictedScribe, probabilities = identifier.predict(testImg, scribeNames)
    
    #print(f"\nPredicted scribe: {predictedScribe}")
    #print("Probabilities:")
    #for scribe, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        #print(f"  {scribe}: {prob:.4f}")