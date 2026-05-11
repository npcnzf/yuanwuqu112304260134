import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageOps
from model import CNNModel, load_model
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel()

# 模型路径 - 尝试不同的可能位置
model_paths = ['model_best.pth', '机械学习实验三/model_best.pth']
model_loaded = False

for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            model = load_model(model, model_path, device)
            print(f"Model loaded successfully from {model_path}!")
            model_loaded = True
            break
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")

if not model_loaded:
    print("Warning: Model file not found. Using untrained model for demo.")

model.to(device)


def preprocess_image(image):
    if isinstance(image, dict):
        image = image.get('composite', None)
    
    if image is None:
        return None
    
    img = Image.fromarray(np.uint8(image)).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np[np.newaxis, np.newaxis, :, :]
    return torch.tensor(img_np, dtype=torch.float32)


def predict_digit(image):
    if image is None:
        return "请上传或绘制一张图片", None, None
    
    img_tensor = preprocess_image(image)
    if img_tensor is None:
        return "图片预处理失败", None, None
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.exp(outputs)
        _, predicted = torch.max(outputs.data, 1)
        top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    digit = predicted.item()
    confidence = probabilities[0][digit].item()
    
    top3 = [(idx.item(), prob.item()) for idx, prob in zip(top3_indices[0], top3_probs[0])]
    top3_str = "\n".join([f"数字 {d}: {p*100:.2f}%" for d, p in top3])
    
    return f"预测数字: {digit}\n置信度: {confidence*100:.2f}%", top3_str, img_tensor.squeeze().cpu().numpy()


def clear_canvas():
    return None, "等待输入...", "", None


with gr.Blocks(title="手写数字识别系统") as demo:
    gr.Markdown("# 手写数字识别系统")
    gr.Markdown("基于CNN的手写数字识别模型,准确率可达0.988+")
    
    with gr.Tab("画板绘制"):
        with gr.Row():
            with gr.Column():
                canvas = gr.Sketchpad(
                    label="在这里绘制数字",
                    height=300,
                    width=300
                )
                with gr.Row():
                    predict_btn = gr.Button("识别", variant="primary")
                    clear_btn = gr.Button("清除")
            
            with gr.Column():
                result_text = gr.Textbox(label="识别结果", placeholder="等待输入...", lines=3)
                top3_text = gr.Textbox(label="Top 3预测", placeholder="", lines=4)
                processed_image = gr.Image(label="预处理后的图片", height=150, width=150)
        
        predict_btn.click(
            fn=predict_digit,
            inputs=[canvas],
            outputs=[result_text, top3_text, processed_image]
        )
        clear_btn.click(
            fn=clear_canvas,
            outputs=[canvas, result_text, top3_text, processed_image]
        )
    
    with gr.Tab("图片上传"):
        with gr.Row():
            with gr.Column():
                image_upload = gr.Image(label="上传图片", type="numpy", height=300, width=300)
                predict_upload_btn = gr.Button("识别", variant="primary")
            
            with gr.Column():
                result_upload = gr.Textbox(label="识别结果", placeholder="等待上传...", lines=3)
                top3_upload = gr.Textbox(label="Top 3预测", placeholder="", lines=4)
                processed_upload = gr.Image(label="预处理后的图片", height=150, width=150)
        
        predict_upload_btn.click(
            fn=predict_digit,
            inputs=[image_upload],
            outputs=[result_upload, top3_upload, processed_upload]
        )
    
    gr.Markdown("""
    ### 使用说明:
    1. 在画板上绘制0-9之间的数字,或者上传一张包含手写数字的图片
    2. 点击"识别"按钮进行预测
    3. 系统将显示预测结果和置信度
    4. 点击"清除"可以重新绘制
    """)


if __name__ == "__main__":
    # 为Render部署配置正确的启动端口
    import os
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
