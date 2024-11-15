import os
import numpy as np
import librosa
import soundfile as sf
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES


class AudioProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensorineural hearing loss sim")
        self.root.geometry("600x650")
        self.audio_paths = []
        self.processed_audios = []
        self.sampling_rates = []
        self.save_directory = None
        self.loading = False

        # Drag and drop label
        self.drop_label = tk.Label(self.root, text="Drag and drop audio files here", relief="solid", pady=10)
        self.drop_label.pack(pady=10, fill="x")
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.handle_file_drop)

        # Upload button
        self.upload_button = tk.Button(self.root, text="Upload Audio Files", command=self.open_files)
        self.upload_button.pack(pady=5)

        # Listbox for selected files
        self.file_listbox = tk.Listbox(self.root, height=10, width=50, selectmode=tk.SINGLE)
        self.file_listbox.pack(pady=10)

        # Process button
        self.process_button = tk.Button(self.root, text="Process Audio", command=self.start_processing_thread)
        self.process_button.pack(pady=5)

        # Choose directory button
        self.choose_dir_button = tk.Button(self.root, text="Choose Save Directory", command=self.choose_save_directory)
        self.choose_dir_button.pack(pady=5)

        # Save button
        self.save_button = tk.Button(self.root, text="Save Processed Audios", command=self.save_audios, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # Canvas for loading animation
        self.canvas = tk.Canvas(self.root, width=300, height=150)
        self.canvas.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="", fg="green")
        self.status_label.pack(pady=10)

    def handle_file_drop(self, event):
        dropped_files = self.parse_file_paths(event.data)
        for file in dropped_files:
            if file not in self.audio_paths and os.path.isfile(file):
                self.audio_paths.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))

    def open_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
        for file in files:
            if file not in self.audio_paths:
                self.audio_paths.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))

    def parse_file_paths(self, file_data):
        return file_data.strip().split()

    def start_loading_animation(self):
        """Starts the loading animation."""
        self.loading = True
        self.animate_loading(0)

    def stop_loading_animation(self):
        """Stops the loading animation."""
        self.loading = False
        self.canvas.delete("all")

    def animate_loading(self, angle):
        """Animates a circular loading indicator."""
        self.canvas.delete("all")
        if self.loading:
            x0, y0, x1, y1 = 120, 50, 180, 110
            extent = 90
            self.canvas.create_arc(x0, y0, x1, y1, start=angle, extent=extent, outline="black", style="arc", width=4)
            self.root.after(50, self.animate_loading, (angle + 10) % 360)

    def start_processing_thread(self):
        """Starts the audio processing in a separate thread."""
        processing_thread = threading.Thread(target=self.process_audios)
        processing_thread.start()

    def process_audios(self):
        if not self.audio_paths:
            messagebox.showerror("Error", "No audio files loaded.")
            return

        self.start_loading_animation()
        self.processed_audios.clear()
        self.sampling_rates.clear()

        try:
            for audio_path in self.audio_paths:
                audio, sr = librosa.load(audio_path, sr=None)

                # Apply frequency-dependent attenuation
                def apply_frequency_dependent_attenuation(signal, sampling_rate):
                    fft_signal = np.fft.rfft(signal)
                    freqs = np.fft.rfftfreq(len(signal), 1 / sampling_rate)
                    attenuation = np.ones_like(freqs)
                    attenuation[freqs > 1000] *= 0.6
                    attenuation[freqs > 3000] *= 0.3
                    attenuation[freqs > 6000] *= 0.1
                    fft_signal = fft_signal * attenuation
                    return np.fft.irfft(fft_signal)

                # Apply dynamic range compression
                def apply_dynamic_range_compression(signal):
                    compressed_signal = np.sign(signal) * np.log1p(np.abs(signal)) * 1.2
                    return compressed_signal / np.max(np.abs(compressed_signal))

                attenuated_audio = apply_frequency_dependent_attenuation(audio, sr)
                processed_audio = apply_dynamic_range_compression(attenuated_audio)
                self.processed_audios.append(processed_audio)
                self.sampling_rates.append(sr)

            self.save_button.config(state=tk.NORMAL)
            self.status_label.config(text="Audio files processed successfully.", fg="green")
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
        finally:
            self.stop_loading_animation()

    def choose_save_directory(self):
        self.save_directory = filedialog.askdirectory()
        if self.save_directory:
            self.status_label.config(text=f"Save directory selected: {self.save_directory}", fg="blue")

    def save_audios(self):
        if not self.processed_audios or not self.sampling_rates:
            messagebox.showerror("Error", "No processed audio to save.")
            return

        if not self.save_directory:
            messagebox.showerror("Error", "No save directory selected.")
            return

        try:
            saved_files = []
            for i, audio_path in enumerate(self.audio_paths):
                base_name, ext = os.path.splitext(os.path.basename(audio_path))
                output_path = os.path.join(self.save_directory, f"{base_name}_SNHL_sim{ext}")
                sf.write(output_path, self.processed_audios[i], self.sampling_rates[i])
                saved_files.append(output_path)

            self.status_label.config(
                text=f"Processed files saved:\n" + "\n".join(saved_files), fg="blue"
            )
            messagebox.showinfo("Success", "All processed files have been saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving the files: {e}")


# Main Application Execution
if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Use TkinterDnD to enable drag-and-drop functionality
    app = AudioProcessingApp(root)
    root.mainloop()
