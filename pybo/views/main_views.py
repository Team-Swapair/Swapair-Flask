from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os

# localhost:5000/
bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/captcha', methods=['POST'])
def predict():
    file = request.files['captchaImg']
    file.save('C:/projects/myproject/pybo/uploads/' + secure_filename(file.filename))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 원본 사진 불러와서 resizing
    img_color = cv2.imread('C:/projects/myproject/pybo/uploads/' + file.filename, 1)
    img_color = cv2.resize(img_color, (1024, 1024))
    # cv2.imshow('color', img_color)

    # 원본 사진 포스티잇 부분 자르기
    output = img_color[512:, :512]

    # cv2.imshow("output", output)

    height, width, channel = output.shape

    # 이미지 grayscale 처리
    img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray", img_gray)

    # 가우시안블러 필터 적용
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # cv2.imshow("gaussian", img_gray)

    ret, img_th = cv2.threshold(img_blur, 160, 255, cv2.THRESH_BINARY_INV)

    edged = cv2.Canny(img_blur, 10, 250)
    # cv2.imshow('Edged', edged)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closed', closed)

    contours, hier = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # contours_image = cv2.drawContours(output, contours, -1, (0,255,0), 3)
    # cv2.imshow('contours_image', contours_image)

    '''
    rect[0]: 직사각형의 왼쪽 상단 점의 x좌표
    rect[1]: 직사각형의 왼쪽 상단 점의 y좌표
    rect[2]: 직사각형의 가로 길이
    rect[3]: 직사각형의 세로 길이
    '''
    rects = [cv2.boundingRect(each) for each in contours]
    rects = sorted(rects)
    # print(rects)

    result = []
    for rect in rects:
        if rect[3] == 73 or rect[3] == 67:
            continue
        if 5 < rect[2] < 80 and 30 < rect[3] < 150:
            result.append(rect)

    # print(result)

    for rect in result:
        print(rect)
        cv2.circle(img_blur, (rect[0], rect[1]), 10, (0, 0, 255), -1)
        cv2.circle(img_blur, (rect[0] + rect[2], rect[1] + rect[3]), 10, (0, 0, 255), -1)
        cv2.rectangle(img_blur, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

        # cv2.imshow('imgblur', img_blur)

    #
    # 이전에 처리해놓은 이미지 사용
    img_for_class = closed.copy()

    # 최종 이미지 파일용 배열
    mnist_imgs = []
    margin_pixel = 15

    # 숫자 영역 추출 및 (28,28,1) reshape

    for rect in result:
        # print(rect)
        # 숫자영역 추출
        im = img_for_class[rect[1] - margin_pixel:rect[1] + rect[3] + margin_pixel,
             rect[0] - margin_pixel:rect[0] + rect[2] + margin_pixel]
        row, col = im.shape[:2]

        # 정방형 비율을 맞춰주기 위해 변수 이용
        bordersize = max(row, col)
        diff = min(row, col)

        # 이미지의 intensity의 평균을 구함
        bottom = im[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        # border추가해 정방형 비율로 보정
        border = cv2.copyMakeBorder(
            im,
            top=0,
            bottom=0,
            left=int((bordersize - diff) / 2),
            right=int((bordersize - diff) / 2),
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )

        square = border
        # cv2.imshow('square', square)

        # square 사이즈 (28,28)로 축소
        resized_img = cv2.resize(square, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        mnist_imgs.append(resized_img)
        # cv2.imshow('resized_img', resized_img)

    resultArr = []

    model = load_model('C:/projects/myproject/pybo/CNN-Model.h5')

    # for i in range(len(mnist_imgs)):
    for i in range(5):

        img = mnist_imgs[i]
        # cv2.imshow('img', img)

        # 이미지를 784개 흑백 픽셀로 사이즈 변환
        img = img.reshape(-1, 28, 28, 1)

        # 데이터를 모델에 적용할 수 있도록 가공
        input_data = ((np.array(img) / 255) - 1) * -1

        # 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
        res = np.argmax(model.predict(input_data), axis=-1)

        resultArr.append(res[0])
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    result = ''.join(map(str, resultArr))

    return jsonify({"result": result})
