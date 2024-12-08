# download necessary NLTK resources
# stopwords: words such as
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

## set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReviewDataset(Dataset):
    """
    Initialize the dataset with reviews, features and labels.


    :param texts: List/array of review texts
    :param app_features: List/array of feature vectors for each app
    :param lables: List/array of corresponding labels/ratings

    """

    def __init__(self, texts, app_features, lables):
        self.texts = texts
        self.app_features = app_features
        self.lables = lables

    """
    Return the total number of samples in the dataset.
    Required by PyTorch Dataset class.

    :returns int: Number of reviews/samples in the dataset

    """

    def __len__(self):
        return len(self.texts)

    """
    Fetch a single sample from the dataset at the specified index.
    Required by PyTorch Dataset class.

    :param idx: Index of the sample to retrieve
    :returns tuple: (review_text, app_features, label) for the specified index
    """

    def __getitem__(self, idx):
        return self.texts[idx], self.app_features[idx], self.lables[idx]


# create LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim, n_layers, dropout, app_feature_dim):
        """
        Initialize LSTM Model for Sentiment Classification with App Features

        :param vocab_size (int): Total number of unique words in vocabulary
        :param embedding_dim (int): Size of word embedding vectors
        :param output_dim (int): Number of output classes (e.g., 2 for binary sentiment)
        :param hidden_dim (int): Number of features in LSTM hidden state
        :param n_layers (int): Number of stacked LSTM layers
        :param dropout (float): Dropout rate for regularization
        :param app_feature_dim (int): Dimension of additional app-specific features
        """
        super().__init__()

        self.vocab_size = vocab_size

        # Create embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer processes sequence of word embeddings
        # batch_first=True means input shape is (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)

        # First fully connected layer combines LSTM output with app features
        self.fc1 = nn.Linear(hidden_dim + app_feature_dim, 128)

        # Output layer produces final classification
        self.fc2 = nn.Linear(128, output_dim)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # ReLU activation function adds non-linearity
        self.relu = nn.ReLU()

    def forward(self, text, app_features):
        """
        Forward pass of the model


        :param text (torch.Tensor): Input tensor of word indices, shape (batch_size, sequence_length)
        :param app_features (torch.Tensor): Additional app-specific features, shape (batch_size, app_feature_dim)


        :returns torch.Tensor: Model predictions, shape (batch_size, output_dim)
        """
        # Convert word indices to embeddings and apply dropout
        embedded = self.dropout(self.embedding(text))  # Shape: (batch_size, seq_len, embedding_dim)

        # Process sequence through LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # Use final hidden state from last LSTM layer
        hidden = self.dropout(hidden[-1, :, :])
        # Concatenate hidden state with app features
        combined = torch.cat([hidden, app_features], dim=1)  # Shape: (batch_size, hidden_dim)

        # Pass through fully connected layers with dropout and ReLU
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)

        # Return final predictions
        return self.fc2(x)


class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis pipeline that processes app reviews
    Combines text processing, LSTM modeling, and app-specific features

    Key Features:
    - Text preprocessing with emoji preservation
    - Custom vocabulary building
    - LSTM-based sentiment classification
    - Integration of app-specific features
    - Built-in training and evaluation pipeline

    :param df: DataFrame containing review data
    :param content_col: Name of column containing review text
    :param label_col: Name of column containing sentiment labels
    :param app_col: Name of column containing app identifiers
    :param test_size: Proportion of data to use for testing (0-1)
    :param random_state: Random seed for reproducible results
    """

    def __init__(self, df: pd.DataFrame, content_col: str = "cleaned_review",
                 label_col: str = "flag", app_col: str = "app",
                 test_size: float = 0.2, random_state: int = 42):

        # Create a copy of input data to avoid modifications to original
        self.df = df.copy()

        # Preprocess all review texts
        self.df['processed_text'] = self.df[content_col].apply(self.preprocess_text)

        # Build vocabulary from processed texts
        self.vocab = self.build_vocab(self.df['processed_text'])

        # Create word-to-index mapping with special tokens
        self.word_to_idx = {word: i + 1 for i, word in enumerate(self.vocab)}
        self.word_to_idx['<PAD>'] = 0
        self.word_to_idx['<UNK>'] = len(self.word_to_idx)

        # Convert texts to numerical sequences
        self.encoded_texts = self.encode_texts(self.df['processed_text'])

        # Encode sentiment labels
        label_encoder = LabelEncoder()
        self.labels = torch.tensor(label_encoder.fit_transform(self.df[label_col]))

        # One-hot encode app features
        app_encoder = OneHotEncoder(sparse_output=False)
        self.app_features = torch.tensor(
            app_encoder.fit_transform(self.df[app_col].values.reshape(-1, 1)),
            dtype=torch.float32
        )

        # Split data into training and test sets
        self.X_train, self.X_test, self.app_train, self.app_test, self.y_train, self.y_test = train_test_split(
            self.encoded_texts,
            self.app_features,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )

        # Create PyTorch datasets
        self.train_dataset = ReviewDataset(self.X_train, self.app_train, self.y_train)
        self.test_dataset = ReviewDataset(self.X_test, self.app_test, self.y_test)

        # Define model hyperparameters
        self.model_params = {
            'vocab_size': len(self.word_to_idx),
            'embedding_dim': 100,
            'hidden_dim': 128,
            'output_dim': len(np.unique(self.labels)),
            'n_layers': 2,
            'dropout': 0.5,
            'app_feature_dim': self.app_features.shape[1]
        }

        # Initialize model, loss function, and optimizer
        self.model = SentimentLSTM(**self.model_params)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text while preserving emojis

        Steps:
        1. Convert to lowercase
        2. Tokenize while keeping emojis intact
        3. Remove stopwords (except emojis)
        4. Join tokens back into text
        """
        # Convert to lowercase
        text = text.lower()

        def tokenize_with_emojis(text):
            """
            Custom tokenizer that preserves emojis as distinct tokens
            Handles the text character by character to properly separate
            emojis from regular words
            """
            tokens = []
            current_token = []

            for char in text:
                if emoji.is_emoji(char):
                    # If there's a current token, save it
                    if current_token:
                        tokens.append(''.join(current_token))
                        current_token = []
                    # Add emoji as a separate token
                    tokens.append(char)
                elif char.isalnum():
                    current_token.append(char)
                else:
                    # If there's a current token, save it
                    if current_token:
                        tokens.append(''.join(current_token))
                        current_token = []

            # Handle any remaining token
            if current_token:
                tokens.append(''.join(current_token))

            return tokens

        # Tokenize and remove stopwords (preserve emojis)
        tokens = tokenize_with_emojis(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words or emoji.is_emoji(token)]
        return ' '.join(tokens)

    def build_vocab(self, texts: pd.Series, max_vocab_size: int = 10000) -> List[str]:
        """
        Build vocabulary from texts, including emojis


        :param texts (pd.Series): Series of processed texts
        :param max_vocab_size (int): Maximum vocabulary size


        :returns List[str]: Vocabulary words and emojis
        """
        # Count word frequencies
        word_freq = {}
        for text in texts:
            for token in text.split():
                word_freq[token] = word_freq.get(token, 0) + 1

        # Sort by frequency and limit vocabulary
        vocab = sorted(word_freq, key=word_freq.get, reverse=True)[:max_vocab_size]
        return vocab

    def encode_texts(self, texts: pd.Series, max_length: int = 100) -> torch.Tensor:
        """
        Encode texts to tensor, handling emojis

        :param texts (pd.Series): Series of processed texts
        :param max_length (int): Maximum sequence length

        :returns torch.Tensor: Encoded texts
        """
        encoded_texts = []
        for text in texts:
            # Convert tokens to indices
            indices = [
                self.word_to_idx.get(token, self.word_to_idx['<UNK>'])
                for token in text.split()
            ]

            # Pad or truncate to fixed length
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [self.word_to_idx['<PAD>']] * (max_length - len(indices))

            encoded_texts.append(indices)

        return torch.tensor(encoded_texts)

    def train(self, epochs: int = 10, batch_size: int = 32):
        """
        Train the LSTM model


        :param epochs (int): Number of training epochs
        :param batch_size (int): Batch size for training
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_total_loss = 0

            for texts, app_features, labels in train_loader:
                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(texts, app_features)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                train_total_loss += loss.item()

            # Calculate average training loss
            avg_train_loss = train_total_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            test_total_loss = 0

            with torch.no_grad():
                for texts, app_features, labels in test_loader:
                    outputs = self.model(texts, app_features)
                    loss = self.criterion(outputs, labels)
                    test_total_loss += loss.item()

            # Calculate average test loss
            avg_test_loss = test_total_loss / len(test_loader)

            # Print both losses
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Test Loss: {avg_test_loss:.4f}\n')

    def evaluate(self) -> dict:
        """
        Evaluate model performance


        :returns dict: Performance metrics
        """
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for texts, app_features, labels in test_loader:
                outputs = self.model(texts, app_features)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'classification_report': classification_report(all_labels, all_preds)
        }

        return metrics

# Check for significant loss increase
# if epoch > 0:
#     loss_increase = (avg_train_loss - self.train_losses[-2]) / self.train_losses[-2]
#     if loss_increase > 0.5:  # 50% increase
#         print(f"\nWarning: Training loss increased by {loss_increase*100:.1f}%")
#         print("Consider adjusting learning rate or gradient clipping threshold")
#
# print("-" * 80)

# if avg_test_loss < best_loss:
#     best_loss = avg_test_loss
#     patience_counter = 0
#     # move back to mps while saveing the model
#     model_state = {k: v.to('cpu') for k, v in self.model.state_dict().items()}
#     torch.save(model_state, 'best_model.pt')
# else:
#     patience_counter += 1
#     if patience_counter >= patience:
#         print("Early stopping triggered")
#         # move back to mps while loading the model
#         state_dict = torch.load('best_model.pt')
#         self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
#         break

