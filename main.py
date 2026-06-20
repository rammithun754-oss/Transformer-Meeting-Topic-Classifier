import torch
import torch.nn as nn
import torch.optim as optim

# ====================================
# 1. Dataset
# ====================================

sentences = [

    "we need to hire engineers",

    "the hiring budget will increase",

    "rahul will contact agencies",

    "the team agreed to buy gpu servers",

    "budget planning meeting",

    "purchase additional infrastructure",

    "recruit three ai engineers",

    "increase hiring budget",

    "contact recruitment agencies",

    "buy additional gpu servers"
]

labels = [

    0,  # Hiring

    1,  # Budget

    2,  # Action

    3,  # Decision

    1,

    4,

    0,

    1,

    2,

    3
]

label_map = {

    0: "Hiring",

    1: "Budget",

    2: "Action",

    3: "Decision",

    4: "Infrastructure"
}

# ====================================
# 2. Tokenization
# ====================================

tokenized_sentences = [

    sentence.lower().split()

    for sentence in sentences
]

# ====================================
# 3. Vocabulary give word a id
# ====================================

vocab = {}

index = 1

for sentence in tokenized_sentences:

    for word in sentence:

        if word not in vocab:

            vocab[word] = index

            index += 1

# ====================================
# 4. Word IDs convert that id to list
# ====================================

sequences = []

for sentence in tokenized_sentences:

    sequence = []

    for word in sentence:

        sequence.append(
            vocab[word]
        )

    sequences.append(
        sequence
    )

# ====================================
# 5. Padding matching all list my adding 0S
# ====================================

max_length = max(

    len(sequence)

    for sequence in sequences
)

padded_sequences = []

for sequence in sequences:

    padded = sequence + [0] * (

        max_length - len(sequence)

    )

    padded_sequences.append(
        padded
    )

# ====================================
# 6. Tensor Conversion
# ====================================

X = torch.tensor(

    padded_sequences,

    dtype=torch.long
)

y = torch.tensor(

    labels,

    dtype=torch.long
)

# ====================================
# 7. Positional Encoding
# ====================================

class PositionalEncoding(

    nn.Module
):

    def __init__(

        self,

        d_model,

        max_len=100 # pepare positions
    ):
        super().__init__()

        pe = torch.zeros(

            max_len,

            d_model
        )# create empty vector spaces in position for each position


        position = torch.arange(

            0,

            max_len

        ).unsqueeze(1) # store each posiyions to each word id

        div_term = torch.exp( # store vector value in  empty place

            torch.arange(
                0,
                d_model,
                2
            )

            * (

                -torch.log(
                    torch.tensor(
                        10000.0
                    )
                )

                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(

            position * div_term
        )

        pe[:, 1::2] = torch.cos(

            position * div_term
        )

        self.register_buffer(

            "pe",

            pe.unsqueeze(0)
        )

    def forward( # vector to word maening

        self,

        x
    ):

        x = x + self.pe[
            :,
            :x.size(1)
        ]

        return x

# ====================================
# 8. Transformer Classifier
# ====================================

class MeetingClassifier(

    nn.Module
):

    def __init__(

        self,

        vocab_size,

        d_model=64,

        nhead=8,

        num_layers=2,

        num_classes=5
    ):

        super().__init__()

        # Embedding

        self.embedding = nn.Embedding(

            vocab_size,

            d_model
        )

        # Positional Encoding

        self.position = PositionalEncoding(

            d_model # meaning plus position
        )

        # Transformer Encoder Layer

        encoder_layer = ( # word relationship

            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                batch_first=True
            )
        )

        # Transformer Encoder

        self.transformer_encoder = (

            nn.TransformerEncoder(

                encoder_layer,

                num_layers=num_layers
            )
        )

        # Classification Head

        self.fc = nn.Linear(

            d_model,

            num_classes
        )

    def forward(

        self,

        x
    ):

        # Embedding

        x = self.embedding(x)

        # Positional Encoding

        x = self.position(x)

        # Transformer Encoder

        x = self.transformer_encoder(x)

        # Pooling

        x = x.mean(dim=1)

        # Classification

        x = self.fc(x)

        return x

# ====================================
# 9. Initialize Model
# ====================================

vocab_size = len(vocab) + 1

model = MeetingClassifier(

    vocab_size=vocab_size,

    d_model=64,

    nhead=8,

    num_layers=2,

    num_classes=5
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001
)

# ====================================
# 10. Training
# ====================================

epochs = 300

for epoch in range(

    epochs
):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(

        outputs,

        y
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:

        print(

            f"Epoch {epoch+1}"

            f" Loss: "

            f"{loss.item():.4f}"
        )

# ====================================
# 11. Prediction Function
# ====================================

def predict_topic(

    sentence
):

    words = sentence.lower().split()

    sequence = []

    for word in words:

        if word in vocab:

            sequence.append(

                vocab[word]
            )

        else:

            sequence.append(0)

    sequence += [0] * (

        max_length - len(sequence)
    )

    tensor = torch.tensor(

        [sequence],

        dtype=torch.long
    )

    with torch.no_grad():

        output = model(

            tensor
        )

        prediction = torch.argmax(

            output,

            dim=1
        ).item()

    print("\nSentence:")

    print(sentence)

    print("\nPrediction:")

    print(

        label_map[
            prediction
        ]
    )

# ====================================
# 12. Testing
# ====================================

predict_topic(

    "we need more engineers"
)

predict_topic(

    "increase the hiring budget"
)

predict_topic(

    "rahul will contact vendors"
)

predict_topic(

    "the team agreed to buy servers"
)

predict_topic(

    "purchase infrastructure equipment"
)
