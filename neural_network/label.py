""""
Project-wide variable to hold the correct label (one hot-encoded).
    -Ideal values hot-coded as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""""
current_label = None

# Update the label
def update_label(label : "np.ndarray"):
    global current_label
    current_label = label
