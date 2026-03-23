
# --- Imports ---
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save plots instead of showing

# --- Step 1: Data Collection ---
# Done individually and on everyone's personal device

# --- Step 2: Data Storage ---

# ** Read CSV Data **
# Jumping Data
brandon_jump_data = pd.read_csv('Brandon/jump.csv', encoding='utf-16')
print(brandon_jump_data.columns)  # <--- Add it here

paolo_jump_data = pd.read_csv('Paolo/jump.csv', encoding='utf-16')
shayan_jump_data = pd.read_csv('Shayan/jump.csv', encoding='utf-16')
brandon_jump_data['Label'] = 1     # Assign label 1 for jumping
paolo_jump_data['Label'] = 1   # Assign label 1 for jumping
shayan_jump_data['Label'] = 1    # Assign label 1 for jumping

# Walking Data
brandon_walk_data = pd.read_csv('Brandon/walk.csv', encoding='utf-16')
paolo_walk_data = pd.read_csv('Paolo/walk.csv', encoding='utf-16')
shayan_walk_data = pd.read_csv('Shayan/walk.csv', encoding='utf-16')
brandon_walk_data['Label'] = 0     # Assign label 0 for walking
paolo_walk_data['Label'] = 0   # Assign label 0 for walking
shayan_walk_data['Label'] = 0    # Assign label 0 for walking

# Removed ordering stuff per request

# ** Segment The Data **
def segment_data(data, chunk_size=500):
    """
    Segments the data into fixed-size chunks.
    Each chunk represents 5-second windows of accelerometer data.
    """
    segments = []
    number_of_segments = len(data) // chunk_size
    for i in range(number_of_segments):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        segment = data.iloc[start_index:end_index]
        segments.append(segment)
    return segments

# Segment walking and jumping data for all
walk_segments_brandon = segment_data(brandon_walk_data)
walk_segments_paolo = segment_data(paolo_walk_data)
walk_segments_shayan = segment_data(shayan_walk_data)

jump_segments_brandon = segment_data(brandon_jump_data)
jump_segments_paolo = segment_data(paolo_jump_data)
jump_segments_shayan = segment_data(shayan_jump_data)

# Combine segmented data
walk_segments = walk_segments_brandon + walk_segments_paolo + walk_segments_shayan
jump_segments = jump_segments_brandon + jump_segments_paolo + jump_segments_shayan
combined_data = walk_segments + jump_segments

# Filter out segments that do not match the expected shape for each group
expected_shape_walk = walk_segments[0].shape
walk_segments = [seg for seg in walk_segments if seg.shape == expected_shape_walk]
expected_shape_jump = jump_segments[0].shape
jump_segments = [seg for seg in jump_segments if seg.shape == expected_shape_jump]

# ** Shuffle the Data **
np.random.seed(42)
np.random.shuffle(combined_data)

# ** Train-Test Split **
# Combine segments into a DataFrame for splitting
combined_df = pd.concat(combined_data, ignore_index=True)
train_data, test_data = train_test_split(combined_df, test_size=0.1, random_state=42, shuffle=True)

# ** Store Data in HDF5 File **
with h5py.File("motion_data.h5", "w") as hdf:
    # Store raw data
    hdf.create_dataset("raw_data/brandon_jumping", data=brandon_jump_data.to_numpy())
    hdf.create_dataset("raw_data/paolo_jumping", data=paolo_jump_data.to_numpy())
    hdf.create_dataset("raw_data/shayan_jumping", data=shayan_jump_data.to_numpy())
    hdf.create_dataset("raw_data/brandon_walking", data=brandon_walk_data.to_numpy())
    hdf.create_dataset("raw_data/paolo_walking", data=paolo_walk_data.to_numpy())
    hdf.create_dataset("raw_data/shayan_walking", data=shayan_walk_data.to_numpy())

    # Store segmented data; np.stack() enforces uniform shape
    hdf.create_dataset("segmented_data/walking", data=np.stack([s.to_numpy() for s in walk_segments]))
    hdf.create_dataset("segmented_data/jumping", data=np.stack([s.to_numpy() for s in jump_segments]))

    # Store train-test split data
    hdf.create_dataset("train_data", data=train_data.to_numpy())
    hdf.create_dataset("test_data", data=test_data.to_numpy())

print("Data successfully stored in motion_data.h5")

# --- Step 3: Data Visualization ---

# 1. Basic Acceleration Graphs

# Brandon - Jumping data, X-axis only
plt.figure(figsize=(8, 4))
plt.plot(brandon_jump_data['Time (s)'], brandon_jump_data['Acceleration x (m/s^2)'], color='blue')
plt.xlabel('Time (s)')
plt.ylabel('X-Axis Acceleration (m/s²)')
plt.title('Brandon - Jumping (X-Axis Only)')
plt.grid(True)
plt.tight_layout()
plt.savefig('brandon_jumping_x.png')
plt.close()

# Paolo - Walking data, Y-axis only
plt.figure(figsize=(8, 4))
plt.plot(paolo_walk_data['Time (s)'], paolo_walk_data['Acceleration y (m/s^2)'], color='green')
plt.xlabel('Time (s)')
plt.ylabel('Y-Axis Acceleration (m/s²)')
plt.title('Paolo - Walking (Y-Axis Only)')
plt.grid(True)
plt.tight_layout()
plt.savefig('paolo_walking_y.png')
plt.close()

# Shayan - Jumping data, Z-axis only
plt.figure(figsize=(8, 4))
plt.plot(shayan_jump_data['Time (s)'], shayan_jump_data['Acceleration z (m/s^2)'], color='red')
plt.xlabel('Time (s)')
plt.ylabel('Z-Axis Acceleration (m/s²)')
plt.title('Shayan - Jumping (Z-Axis Only)')
plt.grid(True)
plt.tight_layout()
plt.savefig('shayan_jumping_z.png')
plt.close()

# 2. Segment 10 Visualization

def plot_segment_10_x(segments_walk, segments_jump, name):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(segments_walk[10]['Time (s)'], segments_walk[10]['Acceleration x (m/s^2)'], label='Walking', color='orange')
    plt.title(f'Segment 10 - Walking ({name})')
    plt.xlabel('Time (s)')
    plt.ylabel('X Accel (m/s²)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(segments_jump[10]['Time (s)'], segments_jump[10]['Acceleration x (m/s^2)'], label='Jumping', color='purple')
    plt.title(f'Segment 10 - Jumping ({name})')
    plt.xlabel('Time (s)')
    plt.ylabel('X Accel (m/s²)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'segment_10_{name.lower()}.png')
    plt.close()

# Segments created in Step 2

plot_segment_10_x(walk_segments_brandon, jump_segments_brandon, "Brandon")
plot_segment_10_x(walk_segments_paolo, jump_segments_paolo, "Paolo")
plot_segment_10_x(walk_segments_shayan, jump_segments_shayan, "Shayan")

# 3. Metadata Plot: Time-Based Recording Durations

brandon_walk = pd.read_csv("Meta Data/brandon_walk_time.csv")
brandon_jump = pd.read_csv("Meta Data/brandon_jump_time.csv")

paolo_walk = pd.read_csv("Meta Data/paolo_walk_time.csv")
paolo_jump = pd.read_csv("Meta Data/paolo_jump_time.csv")

shayan_walk = pd.read_csv("Meta Data/shayan_walk_time.csv")
shayan_jump = pd.read_csv("Meta Data/shayan_jump_time.csv")

# --- Function to Calculate Duration in Seconds ---
def get_duration(df):
    print(df.columns)
    start = pd.to_datetime(df[df['event'] == 'START']["system time text"].iloc[0])
    end = pd.to_datetime(df[df['event'] == 'PAUSE']["system time text"].iloc[0])
    duration = (end - start).total_seconds()
    return duration

# --- Compute Total Durations ---
durations = {
    "Brandon": get_duration(brandon_walk) + get_duration(brandon_jump),
    "Paolo": get_duration(paolo_walk) + get_duration(paolo_jump),
    "Shayan": get_duration(shayan_walk) + get_duration(shayan_jump),
}

plt.figure(figsize=(8, 5))
plt.bar(durations.keys(), durations.values(), color=["blue", "green", "red"])
plt.ylabel("Total Time Collected (seconds)")
plt.title("Total Data Collection Time per Person")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('data_collection_durations.png')
plt.close()

# --- Step 4: Pre-processing ---

# Combine segmented data
all_data_df = pd.concat(combined_data, ignore_index=True)

smoothed_data = all_data_df.rolling(window=5, center=True, min_periods=1).mean()

# Drop NaNs created by smoothing
smoothed_data_clean = smoothed_data.dropna()

# Compute z-scores
sensor_cols = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)',
               'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
sensor_data = smoothed_data_clean[sensor_cols]

z_scores = stats.zscore(sensor_data)
abs_z_scores = np.abs(z_scores)
valid_rows = (abs_z_scores < 3).all(axis=1)

# Filter the smoothed data based on valid z-score rows
filtered_data = smoothed_data_clean[valid_rows]

# Re-segment into 500-sample chunks, ready for model-training
clean_segments = segment_data(filtered_data, chunk_size=500)

# Raw vs. Pre-processed Signal Visualization
# Segment for comparison
seg_index = 42

# Corresponding raw segment
raw_segment = combined_data[seg_index]

# Get indexes for same segment in filtered data
start_index = seg_index * 500
end_index = start_index + 500

# Slicing
clean_segment = filtered_data.iloc[start_index:end_index]

# Plot
plt.figure(figsize=(12, 5))

plt.plot(raw_segment['Time (s)'], raw_segment['Acceleration x (m/s^2)'],
         label='Raw', alpha=0.5, color='blue')

plt.plot(clean_segment['Time (s)'], clean_segment['Acceleration x (m/s^2)'],
         label='Pre-processed', color='orange', linewidth=2)

plt.title('Raw vs. Pre-processed Acceleration (X-Axis, Segment 42)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('raw_vs_preprocessed.png')
plt.close()

#---------Step 5: Feature extraction & normalization-----------
def extract_features(segment):
    feature_list=[]

    for axis in ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']:
        values = segment[axis].values

        feature_list.extend([
            np.max(values),         #Max
            np.min(values),         #Min
            np.max(values) -np.min(values),#Range
            np.mean(values),        #Mean
            np.std(values),         #Standard Deviation
            np.median(values),      #Median
            stats.skew(values),     #Skewness
            stats.kurtosis(values), #Kurtosis
            np.var(values),         #Variance
            np.sqrt(np.mean(values**2)), #RMS
        ])
    return feature_list
#Extract features for training and testing sets
x_train_features =[extract_features(seg) for seg in clean_segments]
x_test_features= [extract_features(seg) for seg in clean_segments]

#convert the feature lists to DataFrames
x_train_df = pd.DataFrame(x_train_features)
x_test_df = pd.DataFrame(x_test_features)

#Retrive labels
y_train =np.array([1 if seg['Label'].mean() >= 0.5 else 0 for seg in clean_segments])
y_test= np.array([1 if seg['Label'].mean() >= 0.5 else 0 for seg in clean_segments])

#Normilize features
scaler=StandardScaler()
x_train_normalized =scaler.fit_transform(x_train_df)
x_test_normalized =scaler.transform(x_test_df)

#------ Step 6: Training and testing a classifier----------------
clf= LogisticRegression(max_iter=10000)
clf.fit(x_train_normalized, y_train)

#Predictions
y_prediction=clf.predict(x_test_normalized)
y_clf_prob=clf.predict_proba(x_test_normalized)
print('Prediction:', y_prediction)
print('Actual:', y_test)
print('Predicted probabilities:', y_clf_prob)

#Evaluation
accuracy= accuracy_score(y_test, y_prediction)
recall= recall_score(y_test, y_prediction)
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')

#Confusion matrix
confused=confusion_matrix(y_test, y_prediction)
ConfusionMatrixDisplay(confused).plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

#ROC
def plot_roc (y_true, y_score, label=1):
    fpr, tpr,_ =roc_curve(y_true, y_score[:, label], pos_label=label)
    roc_auc= roc_auc_score(y_true, y_score[:, label])
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f'ROC Curve (AUC= {roc_auc:.4f})')
    plt.savefig('roc_curve.png')
    plt.close()
plot_roc(y_test,y_clf_prob)

# --- Section 7: Deploying the trained classifier in a Desktop App ---

import tkinter as tk
from tkinter import filedialog, messagebox, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Global variable to store predictions DataFrame for saving later
pred_df_global = None


def classify_from_csv(file_path):
    """
    Reads the CSV, segments it, extracts features, scales them, and returns predictions.
    """
    df_input = pd.read_csv(file_path)
    segments = segment_data(df_input, chunk_size=500)
    features = [extract_features(seg) for seg in segments]
    features_df = pd.DataFrame(features)
    features_scaled = scaler.transform(features_df)
    predictions = clf.predict(features_scaled)
    return predictions


def display_predictions(predictions):
    """
    Displays two plots:
      1) A bar chart showing the distribution of walking vs. jumping.
      2) A scatter plot of predictions vs. segment index.
    The figures have a white background with dark text.
    """
    # --- Bar Chart ---
    walk_count = np.sum(predictions == 0)
    jump_count = np.sum(predictions == 1)

    fig_bar = Figure(figsize=(4, 3), dpi=100, facecolor="white")
    ax_bar = fig_bar.add_subplot(111)
    ax_bar.bar(['Walking', 'Jumping'], [walk_count, jump_count], color=['blue', 'red'])
    ax_bar.set_title('Prediction Distribution', color="black")
    ax_bar.set_xlabel('Activity', color="black")
    ax_bar.set_ylabel('Count', color="black")
    ax_bar.tick_params(axis='x', colors='black')
    ax_bar.tick_params(axis='y', colors='black')
    ax_bar.set_facecolor("white")
    fig_bar.tight_layout()

    # --- Scatter Plot ---
    fig_scatter = Figure(figsize=(4, 3), dpi=100, facecolor="white")
    ax_scatter = fig_scatter.add_subplot(111)
    ax_scatter.scatter(range(len(predictions)), predictions, color='green')
    ax_scatter.set_xlabel('Segment Index', color="black")
    ax_scatter.set_ylabel('Prediction (0=Walk, 1=Jump)', color="black")
    ax_scatter.set_title('Predictions Scatter Plot', color="black")
    ax_scatter.tick_params(axis='x', colors='black')
    ax_scatter.tick_params(axis='y', colors='black')
    ax_scatter.set_facecolor("white")
    fig_scatter.tight_layout()

    # Clear previous content in frames and embed new plots
    for widget in bar_frame.winfo_children():
        widget.destroy()
    for widget in scatter_frame.winfo_children():
        widget.destroy()

    canvas_bar = FigureCanvasTkAgg(fig_bar, master=bar_frame)
    canvas_bar.draw()
    canvas_bar.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    canvas_scatter = FigureCanvasTkAgg(fig_scatter, master=scatter_frame)
    canvas_scatter.draw()
    canvas_scatter.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def save_csv():
    """
    Saves the most recent predictions stored in pred_df_global.
    """
    global pred_df_global
    if pred_df_global is None:
        messagebox.showinfo("No Predictions", "No predictions to save yet!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")])
    if file_path:
        pred_df_global.to_csv(file_path, index=False)
        messagebox.showinfo("Save CSV", f"Predictions saved to {file_path}")


def load_csv():
    """
    Loads a CSV file, classifies it, updates the global predictions DataFrame,
    and displays the results.
    """
    global pred_df_global
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    try:
        predictions = classify_from_csv(file_path)
        messagebox.showinfo("Classification", "Data classified successfully!")
        pred_df_global = pd.DataFrame(predictions, columns=['Predicted Activity'])
        display_predictions(predictions)
    except Exception as e:
        messagebox.showerror("Error", f"Error during classification:\n{e}")


# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Activity Classifier Desktop App")
root.geometry("1000x900")
root.resizable(False, False)  # Lock resizing
root.configure(bg="black")  # Main window background remains black

# Create frames for plots with black background (for overall GUI style)
bar_frame = Frame(root, bg='black', width=400, height=300)
bar_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=20, pady=10)
scatter_frame = Frame(root, bg='black', width=400, height=300)
scatter_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=20, pady=10)

# Button to load CSV and classify (white text on dark background)
load_button = tk.Button(root, text="Load CSV and Classify", font=("Helvetica", 16),
                        command=load_csv, bg="slate blue", fg="white")
load_button.pack(side=tk.BOTTOM, pady=10)

# Single "Save Predictions CSV" button
save_button = tk.Button(root, text="Save Predictions (CSV)", font=("Helvetica", 16),
                        command=save_csv, bg="dark green", fg="white")
save_button.pack(side=tk.BOTTOM, pady=10)

root.mainloop()
