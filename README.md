## How the Project Works (Step-by-Step)

### Part 1: Preparing the Data

* **Step 1: Import Libraries**  
  We load TensorFlow to build the AI model. We use OpenCV to handle images. We use NumPy and Matplotlib for math and charts.

* **Step 2: Set Data Paths**  
  We tell the computer exactly where the X-ray images are stored.

* **Step 3: Clean the Images**  
  We write a tool to fix every image. It turns images black-and-white. It resizes them to 128x128 pixels. It scales pixels between 0 and 1 so the AI works faster.

* **Step 4: Group the Data**  
  We read the CSV text file. We group the children's bone ages into 3 phases: Phase 0, Phase 1, and Phase 2. Then, we clean all images and save them into data lists.



### Part 2: Training the AI

* **Step 5: Split the Data**  
  We divide our data into three separate groups:
  * **80%** to train the AI.
  * **10%** to check the AI during training.
  * **10%** to test the AI at the very end.

* **Step 6: Build the AI Model**  
  We build a simple Convolutional Neural Network (CNN). This AI looks at small parts of the X-ray to find bone shapes and edges.

* **Step 7: Train the AI**  
  We give the AI standard tools (Adam and cross-entropy) to help it learn from its mistakes. We run the training loop multiple times.



### Part 3: Testing and Deployment

* **Step 8: Test the AI**  
  We test the final AI on our unseen test images. This shows us the true accuracy.

* **Step 9: Save the Model**  
  We save the trained AI into a single file named `bone_age_model.h5`.

* **Step 10: Create the Web App**  
  We create a file named `app.py`. This file builds the web page where users can upload X-rays and see the AI results.
