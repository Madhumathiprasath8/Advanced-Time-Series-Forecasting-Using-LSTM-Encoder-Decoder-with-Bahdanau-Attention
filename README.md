**Advanced Time Series Forecasting Using LSTM Encoder–Decoder with Bahdanau Attention**

**1. Introduction**
Time series forecasting plays an essential role in many domains such as energy management, manufacturing, financial markets, and IoT systems. Traditional statistical models like ARIMA and exponential smoothing struggle to capture long-range dependencies and multivariate interactions.
Deep learning architectures — especially LSTM encoder–decoder models enhanced with attention mechanisms — have become powerful solutions for multi-step forecasting.
This project implements:
  •	A multivariate sequence-to-sequence (Seq2Seq) neural network
  •	LSTM encoder-decoder
  •	Bahdanau attention for interpretability and improved accuracy
  •	A baseline persistence model for comparison
The goal is to forecast the next 24 time steps of the target variable using the past 48 time steps of 4 multivariate features.

**2. Dataset Description**
The dataset consists of 1200 time steps with 4 multivariate features:
Feature	Description
target	Main variable to forecast
temp	Environmental temperature
holiday_flag	Binary indicator (0 = normal day, 1 = holiday)
sensor	Additional sensor reading
Dataset Observations:
•	Shape: (1200, 4)
•	No missing values
•	Strong periodicity and smooth trends
•	Suitable for sliding-window deep learning models
Example rows:
time   target   temp   holiday_flag   sensor
0      10.24    22.17       1.0       -0.72
1      11.31    22.42       1.0        0.17
2      12.99    22.94       1.0       -0.54

**3. Problem Formulation**
The forecasting problem is defined as:
* Input: Past 48 time steps of all 4 features
* Output: Future 24 time steps of the target variable
* Task type: Multi-step, multi-feature-to-single-target forecasting
* Model type: Seq2Seq with attention
This structure allows the model to capture short-term & long-term dependencies.

**4. Data Preprocessing**
4.1 Scaling
•	MinMax scaling applied (0–1 range)
•	Ensures stable LSTM training
4.2 Sliding Window Creation
Windows were generated as:
•	Encoder input: (48, 4)
•	Decoder output: (24, 1)
Resulting shapes:
X_enc shape: (1129, 48, 4)
y_dec shape: (1129, 24, 1)
4.3 Data Splitting
Chronological split (no shuffling):
•	Train: 790
•	Validation: 169
•	Test: 170
This avoids leakage from future data.

**5. Baseline Model: Persistence Forecast**
The persistence model predicts:
“The next value = last observed value.”
Baseline Forecast Results
•	MAE: 4.1686
•	RMSE: 5.1006
•	MAPE: 20.45%
This provides a benchmark to beat.

**6. Model Architecture: LSTM Encoder–Decoder with Bahdanau Attention**
6.1 Encoder
•	LSTM(64 units)
•	Encodes last 48 time steps into hidden + cell states
6.2 Decoder
•	LSTM(64 units)
•	Produces outputs step-by-step for 24 timesteps
6.3 Bahdanau Attention
Adds interpretability by learning:
•	Which encoder timesteps are important
•	Producing attention weights of shape (24, 48)
6.4 Final Dense Layers
A TimeDistributed(Dense(1)) generates the prediction sequence.
Total Parameters
49,170 trainable parameters

**7. Model Training**
•	Epochs: 80
•	EarlyStopping applied
•	Loss decreased steadily
•	No overfitting observed
•	Training stable
Sample training loss improvement:
Epoch 1: loss 0.0591 → val_loss 0.0591
Epoch 5: loss 0.0052 → val_loss 0.0055
Epoch 12: loss 0.0015 → val_loss 0.0015
Epoch 32: loss 0.0010 → val_loss 0.0015

**8. Model Evaluation**
After training, the model achieved:
Metric	Value
MAE	0.7446
RMSE	0.9669
MAPE	3.69%
Performance Improvement vs Baseline
•	82% lower MAE
•	81% lower RMSE
•	~83% lower MAPE
This shows substantial forecasting skill.

**9. Attention Interpretation**
The exact Bahdanau attention weights were extracted.
Key Insight:
For every future timestep (t+1 to t+24):
The model consistently focuses on encoder positions:
[47, 46, 45]
Meaning:
•	The last 3 past timesteps influence future predictions the most.
•	This matches real-world intuition: recent history is critical.
Attention Map Shape
(24 decoder steps, 48 encoder steps)
Interpretation
•	Strong peaks near the end of the encoder window
•	Model learns short-term temporal dependencies
•	Holiday flag and temp help model adjust prediction trends

**10. Final Results Summary****
Model	MAE	RMSE	MAPE (%)
Baseline	4.1686	5.1006	20.45
LSTM + Bahdanau Attention	0.7446	0.9669	3.69
The LSTM + Attention model outperforms the baseline with a very large margin.

**11. Conclusion**
The project successfully implemented an advanced deep learning forecasting model with:
•	Multivariate input
•	Sequence-to-sequence learning
•	Bahdanau attention
•	Multi-step forecasting
•	Full interpretability
The model demonstrates excellent accuracy, robustness, and interpretability.

**12. Future Improvements**
Recommended next steps:
•	Try Transformer-based forecasting models
•	Train with teacher forcing in decoder
•	Add SARIMAX, Prophet, or XGBoost as additional baselines
•	Deploy as an API using FastAPI / Flask
•	Convert to ONNX for real-time inference

