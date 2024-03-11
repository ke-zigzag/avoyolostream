#必要なライブラリーのインストール
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import streamlit as st


#モデルインスタンスを作成し、重みを読み込む
model = YOLO("lastver2.pt")

#streamlitのUI
st.title("Avocado checker")
upload_file = st.file_uploader("Please check your avocado here!", type=["jpg", "png"])

if upload_file is not None:
    # アップロードされた画像を表示
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded image", width=250)

    # Check Avocadoボタン
    if st.button('Check Avocado'):
        try:
            # 物体検出の実行
            results = model(image, conf=0.8)

            # 検出結果を処理
            for result in results:
                draw = ImageDraw.Draw(image)
                for box in result.boxes:
                    # バウンディングボックスの座標
                    xyxy = box.xyxy.tolist()
                    xmin, ymin, xmax, ymax = map(int, xyxy[0])

                    # クラスIDと信用度
                    cls_id = int(box.cls.item())
                    label = result.names[cls_id]
                    confidence = box.conf.item()

                    # バウンディングボックスを描画
                    color = (255, 0, 0) if label == 'unripe' else (0, 255, 0)
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

                    # テキストを描画してクラスラベル名と信用度を表示
                    text = f"{label}: {confidence:.2f}"
                    font = ImageFont.load_default(24) #フォントサイズを24に設定
                    draw.text((xmin, ymin), text, align='center', fill=color, font=font)

            # 結果の画像を表示
            st.image(image, caption="Finished detection!", use_column_width=True)
        except Exception as e:
            st.error(f"Error. Please try again...: {str(e)}")