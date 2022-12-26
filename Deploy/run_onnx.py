import os
import numpy as np
import cv2
import onnxruntime


if __name__ == "__main__":
    spath = lambda path: os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

    ort_session = onnxruntime.InferenceSession(spath("srcnn.onnx"))

    input_img = cv2.imread(spath('face.png')).astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    
    ort_inputs = {'input': input_img}
    ort_output = ort_session.run(['output'], ort_inputs)[0]

    ort_output = np.squeeze(ort_output, 0)
    ort_output = np.clip(ort_output, 0, 255)
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
    cv2.imwrite(spath("face_ort.png"), ort_output)