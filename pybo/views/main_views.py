from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

import os

# localhost:5000/
bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/captcha', methods=['POST'])
def predict():
    file = request.files['captchaImg']
    file.save('C:/projects/myproject/pybo/uploads/' + secure_filename(file.filename))

    model = tf.keras.models.load_model('C:/projects/myproject/pybo/suhyun.h5')
    image = cv2.imread('C:/projects/myproject/pybo/uploads/' + file.filename, 1)

    img_color = cv2.resize(image, (1024, 1024))
    # cv2.imshow('color', img_color)
    #
    # 원본 사진 포스티잇 부분 자르기
    image = img_color[650:950, 70:512]
    # cv2.imshow("output", image)

    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 145, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    preprocessed_digits = []

    width = []
    height = []
    xNum = []
    result = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        width.append(w)
        height.append(h)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y + h, x:x + w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        xNum.append(x)
        preprocessed_digits.append(padded_digit)

    df = pd.DataFrame({"w": width, "h": height})
    q3 = df.quantile(0.75)
    q1 = df.quantile(0.25)
    iqr = q3 - q1

    length = len(preprocessed_digits)
    minusIndex = []

    multiple = 0.2

    # for i in range(length):
    #     if width[i] <= q1['w'] - multiple*iqr["w"]:
    #         if width[i] <= q1['h'] - multiple*iqr["h"]:
    #             minusIndex.append(i)

    for i in range(length):
        if width[i] <= 5:
            if width[i] <= 5:
                minusIndex.append(i)

    minusIndex.sort(reverse=True)
    for i in minusIndex:
        del preprocessed_digits[i]
        del xNum[i]

    result = sorted(zip(xNum, preprocessed_digits))
    unzip_numbers, unzip_digits = zip(*result)

    # def is_kor_outlier(df):
    #     wid = df['w']
    #     hei = df['h']
    #     if wid < iqr or hei < iqr:
    #         return True
    #     else:
    #         return False
    #
    # if(is_kor_outlier(df))
    #
    # print("\n\n\n----------------Contoured Image--------------------")
    # plt.imshow(image, cmap="gray")
    # plt.show()

    inp = np.array(preprocessed_digits)

    realResult = []
    for digit in unzip_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))

        # print("\n\n---------------------------------------\n\n")
        # print("=========PREDICTION============ \n\n")
        # plt.imshow(digit.reshape(28, 28), cmap="gray")
        # plt.show()
        # print("\n\nFinal Output: {}".format(np.argmax(prediction)))
        realResult.append(np.argmax(prediction))
        #
        # print("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
        #
        # hard_maxed_prediction = np.zeros(prediction.shape)
        # hard_maxed_prediction[0][np.argmax(prediction)] = 1
        # print("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
        # print("\n\n---------------------------------------\n\n")

    result = ''.join(map(str, realResult))

    return jsonify({"result": result})


@bp.route('/imageCompare', methods=['POST'])
def compare():
    file = request.files['haveImg']
    file.save('C:/projects/myproject/pybo/uploads/' + secure_filename(file.filename))


    fileRoutes = request.form.get('fileRoutes')

    # 원본 사진 불러와서 resizing
    path1 = 'C:/projects/myproject/pybo/uploads/' + file.filename

    for path2 in fileRoutes:
        if check_accuracy(path1, path2):
            res = True
            return jsonify({"result": res})


def check_accuracy(path2, path1):
    # print(path1 +path2)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # cv2.imshow('img',img2)

    h1, w1, c = img1.shape
    h2, w2, c = img2.shape

    if (h1 > h2 / 5) and h1 != h2 and w1 != w2:
        img1 = cv2.resize(img1, (h2, w2))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB, BF-Hamming 로 knnMatch  ---①
    detector = cv2.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x: x.distance)
    # 모든 매칭점 그리기 ---④
    res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow('Matching-All', res1)
    # cv2.waitKey()

    # 매칭점으로 원근 변환 및 영역 표시 ---⑤
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    # RANSAC으로 변환 행렬 근사 계산 ---⑥
    mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape[:2]
    pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
    dst = cv2.perspectiveTransform(pts, mtrx)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # 정상치 매칭만 그리기 ---⑦
    matchesMask = mask.ravel().tolist()
    res2 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                           matchesMask=matchesMask,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # 모든 매칭점과 정상치 비율 ---⑧
    accuracy = float(mask.sum()) / mask.size
    print("match : %d" % (mask.sum()))

    # 결과 출력
    cv2.imshow('Matching-All', res1)
    cv2.imshow('Matching-Inlier ', res2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    if mask.sum() > 100:
        return True
    else:
        return False
