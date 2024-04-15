import requests

def download_file(url, filename):
    # Gửi một HTTP request đến URL
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        # Mở một file mới trong chế độ write binary
        with open(filename, 'wb') as f:
            f.write(r.content)
        print("File đã được tải về thành công!")
    else:
        print("Không thể tải file, mã lỗi:", r.status_code)

# URL của file .caffemodel bạn muốn tải
url = "https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel?raw=true"
# Tên file lưu trữ trên máy local
filename = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

download_file(url, filename)
