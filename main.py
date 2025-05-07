import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import detect
import cv2

class OilSpillApp:
    def __init__(self, root):
        self.root = root
        self.root.minsize(400, 400)
        self.root.title("Обнаружение разливов нефти")
        self.image_path = None
        self.result_img = None

        top_frame = tk.Frame(root)
        top_frame.pack(pady=10, padx=10, fill="x")

        self.path_label = tk.Label(top_frame, text="Путь к файлу:")
        self.path_label.pack(anchor="w")

        self.path_field = tk.Text(top_frame, width=60, height=1, state="disabled",
                                  font=("Segoe UI", 9), bg="white", cursor="arrow")
        self.path_field.pack(fill="x")
        self.path_field.bind("<Button-1>", lambda e: "break")

        self.select_button = tk.Button(top_frame, text="Выбрать изображение", command=self.select_image)
        self.select_button.pack(pady=(5, 0), anchor="w")

        self.image_panel = tk.Label(root)
        self.image_panel.pack(padx=10, pady=20)

        bottom_frame = tk.Frame(root)
        bottom_frame.pack(pady=10)

        self.analyze_button = tk.Button(bottom_frame, text="Анализировать", command=self.analyze_image)
        self.analyze_button.pack(side="left", padx=10)

        self.save_button = tk.Button(bottom_frame, text="Сохранить результат", command=self.save_result, state="disabled")
        self.save_button.pack(side="left", padx=10)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path = path
            self.set_path_text(path)
            img = Image.open(path).resize((500, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=img_tk)
            self.image_panel.image = img_tk
            self.save_button.config(state="disabled")

    def set_path_text(self, text):
        self.path_field.config(state="normal")
        self.path_field.delete("1.0", tk.END)
        self.path_field.insert(tk.END, text)
        self.path_field.config(state="disabled")

    def analyze_image(self):
        if not self.image_path:
            messagebox.showwarning("Внимание", "Сначала выберите изображение.")
            return
        try:
            self.result_img = detect.analyze_save(self.image_path)
            result_pil = Image.fromarray(cv2.cvtColor(self.result_img, cv2.COLOR_BGR2RGB)).resize((500, 300))
            result_tk = ImageTk.PhotoImage(result_pil)
            self.image_panel.configure(image=result_tk)
            self.image_panel.image = result_tk
            self.save_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_result(self):
        if self.result_img is None:
            messagebox.showwarning("Нет результата", "Сначала проанализируйте изображение.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile="result",
                                                 filetypes=[("Изображения JPEG", "*.jpg"), ("Изображения PNG", "*.png")])
        if save_path:
            try:
                cv2.imwrite(save_path, self.result_img)
                messagebox.showinfo("Сохранено", f"Результат сохранён в:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Ошибка при сохранении", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = OilSpillApp(root)
    root.mainloop()
