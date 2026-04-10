import threading
import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image
from pathlib import Path
import torch
import time

from modules.inpainting import SDInpaintingModel, build_pipeline

PREVIEW_W = 512
PREVIEW_H = 512

THUMB_W = 256   # thumbnail panel width
THUMB_H = 256   # thumbnail panel height


class GUI:

    def __init__(self):
        self.state = {
            "image_path": None,
            "mask_path":  None,
            "pipeline":   None,
            "running":    False,
        }

        self.image_np   = np.full((PREVIEW_H, PREVIEW_W, 3), 0.15, dtype=np.float32)
        self.buf_image  = np.full((PREVIEW_W * PREVIEW_H * 3,), 0.15, dtype=np.float32)
        self.buf_output = np.full((PREVIEW_W * PREVIEW_H * 3,), 0.15, dtype=np.float32)
        self.buf_thumb  = np.full((THUMB_W * THUMB_H * 3,), 0.15, dtype=np.float32)
        self.mask_np    = np.zeros((PREVIEW_H, PREVIEW_W), dtype=np.float32)

        self.draw_mode    = "brush"
        self.erase_mode   = False
        self.brush_radius = 20
        self.rect_start   = None

        # Crop state — stored as pixel coords on the THUMBNAIL
        # crop_box = (tx, ty, tsize) where tsize is the side length in thumbnail pixels
        self.orig_image_pil = None   # full-res original PIL image
        self.crop_box       = None   # (tx, ty, tsize) in thumbnail coords
        self.crop_dragging  = False  # dragging the crop box
        self.crop_drag_off  = (0, 0) # offset from box top-left when drag started

        self.output_dir = "./outputs"

    # ------------------------------------------------------------------
    # Composite mask onto canvas image
    # ------------------------------------------------------------------

    def _composite(self):
        img = self.image_np.copy()
        m = self.mask_np > 0
        img[m, 0] = np.clip(img[m, 0] * 0.5 + 0.5, 0.0, 1.0)
        img[m, 1] = img[m, 1] * 0.5
        img[m, 2] = img[m, 2] * 0.5
        np.copyto(self.buf_image, img.flatten())
        dpg.set_value("tex_image", self.buf_image)

    def _pil_to_hwc(self, img: Image.Image, w=PREVIEW_W, h=PREVIEW_H) -> np.ndarray:
        img = img.convert("RGB").resize((w, h), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    # ------------------------------------------------------------------
    # Thumbnail helpers
    # ------------------------------------------------------------------

    def _update_thumb(self):
        """Redraw thumbnail texture and crop box overlay."""
        if self.orig_image_pil is None:
            return
        # Fit original image into THUMB_W x THUMB_H letterboxed
        orig_w, orig_h = self.orig_image_pil.size
        scale = min(THUMB_W / orig_w, THUMB_H / orig_h)
        fit_w = int(orig_w * scale)
        fit_h = int(orig_h * scale)
        off_x = (THUMB_W - fit_w) // 2
        off_y = (THUMB_H - fit_h) // 2

        thumb_img = np.full((THUMB_H, THUMB_W, 3), 0.1, dtype=np.float32)
        resized = self._pil_to_hwc(self.orig_image_pil, fit_w, fit_h)
        thumb_img[off_y:off_y+fit_h, off_x:off_x+fit_w] = resized
        np.copyto(self.buf_thumb, thumb_img.flatten())
        dpg.set_value("tex_thumb", self.buf_thumb)

        # Store fit info for coord mapping
        self._thumb_fit = (off_x, off_y, fit_w, fit_h, orig_w, orig_h, scale)

        # Init crop box if not set — default to largest square centred in fit area
        if self.crop_box is None:
            tsize = min(fit_w, fit_h)
            tx = off_x + (fit_w - tsize) // 2
            ty = off_y + (fit_h - tsize) // 2
            self.crop_box = [tx, ty, tsize]

        self._redraw_crop_box()

    def _redraw_crop_box(self):
        if dpg.does_item_exist("crop_rect"):
            dpg.delete_item("crop_rect")
        if dpg.does_item_exist("crop_handles"):
            dpg.delete_item("crop_handles")
        if self.crop_box is None:
            return
        tx, ty, tsize = self.crop_box
        dpg.draw_rectangle(pmin=(tx, ty), pmax=(tx+tsize, ty+tsize),
                           color=(255, 220, 0, 220), thickness=2,
                           parent="thumb_canvas", tag="crop_rect")

    def _thumb_local_pos(self):
        mouse_pos = dpg.get_mouse_pos(local=False)
        rect_min  = dpg.get_item_state("thumb_canvas").get("rect_min")
        if rect_min is None:
            return -1, -1
        return int(mouse_pos[0] - rect_min[0]), int(mouse_pos[1] - rect_min[1])

    def _is_over_thumb(self):
        x, y = self._thumb_local_pos()
        return 0 <= x < THUMB_W and 0 <= y < THUMB_H

    def _is_over_crop_box(self, x, y):
        if self.crop_box is None:
            return False
        tx, ty, tsize = self.crop_box
        return tx <= x <= tx + tsize and ty <= y <= ty + tsize

    def _clamp_crop_box(self):
        """Keep crop box within the fitted image area."""
        if self.crop_box is None or not hasattr(self, "_thumb_fit"):
            return
        off_x, off_y, fit_w, fit_h, *_ = self._thumb_fit
        tx, ty, tsize = self.crop_box
        tx = max(off_x, min(tx, off_x + fit_w - tsize))
        ty = max(off_y, min(ty, off_y + fit_h - tsize))
        self.crop_box = [tx, ty, tsize]

    def _crop_to_orig_coords(self):
        """Convert crop box thumbnail coords → original image pixel rect."""
        if self.crop_box is None or not hasattr(self, "_thumb_fit"):
            return None
        off_x, off_y, fit_w, fit_h, orig_w, orig_h, scale = self._thumb_fit
        tx, ty, tsize = self.crop_box
        # Map from thumb coords back to original image coords
        ox = int((tx - off_x) / scale)
        oy = int((ty - off_y) / scale)
        osize = int(tsize / scale)
        ox = max(0, min(ox, orig_w - osize))
        oy = max(0, min(oy, orig_h - osize))
        return ox, oy, osize

    def _get_cropped_image(self) -> Image.Image:
        """Return the crop region of the original image as a PIL image."""
        if self.orig_image_pil is None:
            return None
        coords = self._crop_to_orig_coords()
        if coords is None:
            return self.orig_image_pil
        ox, oy, osize = coords
        return self.orig_image_pil.crop((ox, oy, ox + osize, oy + osize))

    def _load_crop_into_canvas(self):
        """Load the current crop region into the canvas + reset mask."""
        cropped = self._get_cropped_image()
        if cropped is None:
            return
        self.image_np = self._pil_to_hwc(cropped)
        self.mask_np[:] = 0.0
        self._composite()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def set_status(self, msg: str, error: bool = False):
        dpg.set_value("status_text", msg)
        color = (220, 80, 80, 255) if error else (100, 220, 100, 255)
        dpg.configure_item("status_text", color=color)

    # ------------------------------------------------------------------
    # Canvas coords
    # ------------------------------------------------------------------

    def _canvas_local_pos(self):
        mouse_pos = dpg.get_mouse_pos(local=False)
        rect_min  = dpg.get_item_state("mask_canvas_win").get("rect_min")
        if rect_min is None:
            return -1, -1
        return int(mouse_pos[0] - rect_min[0]), int(mouse_pos[1] - rect_min[1])

    def _is_over_canvas(self):
        x, y = self._canvas_local_pos()
        return 0 <= x < PREVIEW_W and 0 <= y < PREVIEW_H

    def _clamp_pos(self, x, y):
        return max(0, min(PREVIEW_W - 1, x)), max(0, min(PREVIEW_H - 1, y))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def _paint_brush(self, x, y):
        r    = self.brush_radius
        Y, X = np.ogrid[:PREVIEW_H, :PREVIEW_W]
        self.mask_np[(X - x)**2 + (Y - y)**2 <= r**2] = 0.0 if self.erase_mode else 1.0
        self._composite()

    def _paint_rect(self, x0, y0, x1, y1):
        lx, rx = sorted([x0, x1])
        ty, by = sorted([y0, y1])
        lx, ty = self._clamp_pos(lx, ty)
        rx, by = self._clamp_pos(rx, by)
        self.mask_np[ty:by+1, lx:rx+1] = 0.0 if self.erase_mode else 1.0
        self._composite()

    def _redraw_cursor(self):
        if dpg.does_item_exist("cursor_circle"):
            dpg.delete_item("cursor_circle")
        if not self._is_over_canvas():
            return
        mx, my = self._canvas_local_pos()
        color = (80, 180, 255, 220) if self.erase_mode else (255, 80, 80, 220)
        if self.draw_mode == "brush":
            dpg.draw_circle(center=(mx, my), radius=self.brush_radius,
                            color=color, thickness=2,
                            parent="mask_canvas_win", tag="cursor_circle")
        elif self.draw_mode == "rect":
            if self.rect_start:
                x0, y0 = self.rect_start
                dpg.draw_rectangle(pmin=(x0, y0), pmax=(mx, my),
                                   color=color, thickness=2,
                                   parent="mask_canvas_win", tag="cursor_circle")
            else:
                s = 8
                dpg.draw_line((mx-s, my), (mx+s, my), color=color, thickness=1,
                              parent="mask_canvas_win", tag="cursor_circle")

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def on_mouse_down(self, sender, app_data):
        btn = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
        if btn != 0:
            return

        # Crop box drag on thumbnail
        if self._is_over_thumb() and self.crop_box is not None:
            tx, ty = self._thumb_local_pos()

            if self._is_over_crop_box(tx, ty):
                self.crop_dragging = True
                self.crop_drag_off = (tx - self.crop_box[0], ty - self.crop_box[1])
                return
        # Brush on canvas
        if self._is_over_canvas() and self.draw_mode == "brush":
            x, y = self._canvas_local_pos()
            self._paint_brush(x, y)

    def on_mouse_click(self, sender, app_data):
        btn = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
        if btn != 0:
            return
        
        
        if self._is_over_canvas() and self.draw_mode == "rect":
            x, y = self._canvas_local_pos()
            if self.rect_start is None:
                self.rect_start = (x, y)
            else:
                self._paint_rect(*self.rect_start, x, y)
                self.rect_start = None

    def on_mouse_move(self, sender, app_data):
        self._redraw_cursor()
        # Drag crop box
        if self.crop_dragging and dpg.is_mouse_button_down(0):
            tx, ty = self._thumb_local_pos()
            self.crop_box[0] = tx - self.crop_drag_off[0]
            self.crop_box[1] = ty - self.crop_drag_off[1]
            self._clamp_crop_box()
            self._redraw_crop_box()
            self._load_crop_into_canvas()
            return
        # Brush drag on canvas
        if dpg.is_mouse_button_down(0) and self._is_over_canvas() and self.draw_mode == "brush":
            x, y = self._canvas_local_pos()
            self._paint_brush(x, y)

    def on_mouse_release(self, sender, app_data):
        btn = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
        if btn != 0:
            return
        self.crop_dragging = False

    # ------------------------------------------------------------------
    # Mask actions
    # ------------------------------------------------------------------

    def clear_mask(self):
        self.mask_np[:] = 0.0
        self.rect_start = None
        np.copyto(self.buf_image, self.image_np.flatten())
        dpg.set_value("tex_image", self.buf_image)
        if dpg.does_item_exist("cursor_circle"):
            dpg.delete_item("cursor_circle")
        self.state["mask_path"] = None
        dpg.set_value("mask_path_text", "")

    def save_mask_to_disk(self):
        print(f"[DEBUG] mask max={self.mask_np.max():.3f} sum={self.mask_np.sum():.0f}")
        mask_img = Image.fromarray((self.mask_np * 255).astype(np.uint8), mode="L")
        out = Path(f"{self.output_dir}/mask-{time.time()}.png")
        mask_img.save(out)
        self.state["mask_path"] = str(out)
        self.set_status(f"Mask saved → {out.resolve()}")

    def on_brush_radius_changed(self, sender, app_data):
        self.brush_radius = app_data

    def on_erase_toggle(self, sender, app_data):
        self.erase_mode = app_data

    def on_draw_mode_changed(self, sender, app_data):
        self.draw_mode = app_data
        self.rect_start = None
        dpg.configure_item("brush_radius_slider", show=(app_data == "brush"))

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def on_image_selected(self, sender, app_data, user_data=None):
        path = app_data["file_path_name"]
        self.state["image_path"] = path
        dpg.set_value("image_path_text", path)
        self.orig_image_pil = Image.open(path).convert("RGB")
        self.crop_box = None  # reset crop on new image
        self._update_thumb()
        self._load_crop_into_canvas()

    def on_mask_selected(self, sender, app_data, user_data=None):
        path = app_data["file_path_name"]
        self.state["mask_path"] = path
        img = Image.open(path).convert("L").resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        self.mask_np = np.array(img, dtype=np.float32) / 255.0
        self._composite()
        dpg.set_value("mask_path_text", path)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def build_ui(self):

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(PREVIEW_W, PREVIEW_H, self.buf_image,
                                format=dpg.mvFormat_Float_rgb, tag="tex_image")
            dpg.add_raw_texture(PREVIEW_W, PREVIEW_H, self.buf_output,
                                format=dpg.mvFormat_Float_rgb, tag="tex_output")
            dpg.add_raw_texture(THUMB_W, THUMB_H, self.buf_thumb,
                                format=dpg.mvFormat_Float_rgb, tag="tex_thumb")

        dpg.add_file_dialog(tag="dlg_image", show=False,
                            callback=self.on_image_selected, width=700, height=450)
        dpg.add_file_extension("Images{.png,.jpg,.jpeg}", parent="dlg_image")

        dpg.add_file_dialog(tag="dlg_mask", show=False,
                            callback=self.on_mask_selected, width=700, height=450)
        dpg.add_file_extension("Images{.png,.jpg,.jpeg}", parent="dlg_mask")

        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.on_mouse_down)
            dpg.add_mouse_click_handler(callback=self.on_mouse_click)
            dpg.add_mouse_move_handler(callback=self.on_mouse_move)
            dpg.add_mouse_release_handler(callback=self.on_mouse_release)

        with dpg.window(tag="main_window", label="SD Inpainting"):

            with dpg.collapsing_header(label="Model", default_open=True):
                dpg.add_button(tag="load_button", label="Load Model", callback=self.load_model)

            dpg.add_separator()

            with dpg.collapsing_header(label="Inputs", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Select Image", callback=lambda: dpg.show_item("dlg_image"))
                    dpg.add_text("", tag="image_path_text")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Load Mask from File", callback=lambda: dpg.show_item("dlg_mask"))
                    dpg.add_text("", tag="mask_path_text")

            dpg.add_separator()

            with dpg.collapsing_header(label="Prompt", default_open=True):
                dpg.add_input_text(tag="prompt_input", label="Prompt",
                                   default_value="a red bow accessory on a cosplay costume",
                                   width=600, height=60, multiline=True)
                dpg.add_input_text(tag="neg_prompt_input", label="Negative prompt",
                                   default_value="blurry, low quality, deformed", width=600)

            dpg.add_separator()

            with dpg.collapsing_header(label="Parameters", default_open=True):
                dpg.add_slider_int(tag="steps_slider", label="Steps",
                                   default_value=30, min_value=10, max_value=100, width=300)
                dpg.add_slider_float(tag="guidance_slider", label="Guidance scale",
                                     default_value=7.5, min_value=1.0, max_value=20.0, width=300)
                dpg.add_slider_float(tag="strength_slider", label="Strength",
                                     default_value=0.99, min_value=0.1, max_value=1.0, width=300)

            dpg.add_separator()

            dpg.add_button(tag="run_button", label="Run Inpainting",
                           callback=self.run_inference, enabled=False)

            dpg.add_separator()

            with dpg.group(horizontal=True):

                # LEFT: thumbnail + crop selector
                with dpg.group():
                    dpg.add_text("Crop Region  (drag yellow box)")
                    with dpg.drawlist(width=THUMB_W, height=THUMB_H, tag="thumb_canvas"):
                        dpg.draw_image("tex_thumb", pmin=(0, 0), pmax=(THUMB_W, THUMB_H))

                dpg.add_spacer(width=16)

                # MIDDLE: mask canvas
                with dpg.group():
                    dpg.add_text("Input Image  (paint mask here)")
                    with dpg.drawlist(width=PREVIEW_W, height=PREVIEW_H, tag="mask_canvas_win"):
                        dpg.draw_image("tex_image", pmin=(0, 0), pmax=(PREVIEW_W, PREVIEW_H), tag="canvas_img")

                    with dpg.group(horizontal=True):
                        dpg.add_radio_button(items=["brush", "rect"],
                                             tag="draw_mode_radio", default_value="brush",
                                             horizontal=True, callback=self.on_draw_mode_changed)
                        dpg.add_spacer(width=12)
                        dpg.add_slider_int(tag="brush_radius_slider", label="Brush size",
                                           default_value=20, min_value=5, max_value=80, width=140,
                                           callback=self.on_brush_radius_changed)
                        dpg.add_spacer(width=12)
                        dpg.add_checkbox(label="Erase", tag="erase_checkbox",
                                         callback=self.on_erase_toggle)
                        dpg.add_spacer(width=12)
                        dpg.add_button(label="Clear", callback=self.clear_mask)
                        dpg.add_button(label="Save Mask", callback=self.save_mask_to_disk)

                dpg.add_spacer(width=16)

                # RIGHT: output
                with dpg.group():
                    dpg.add_text("Output")
                    dpg.add_image("tex_output", tag="img_output",
                                  width=PREVIEW_W, height=PREVIEW_H)

            dpg.add_separator()
            dpg.add_text("Ready.", tag="status_text", color=(180, 180, 180, 255))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def load_model(self, sender=None, app_data=None, user_data=None):
        dpg.configure_item("load_button", enabled=False, label="Loading...")
        self.set_status("Loading model weights — this may take a minute...")

        def _worker():
            try:
                model = SDInpaintingModel(device="cuda", dtype=torch.float16)
                model.eval_mode()
                self.state["pipeline"] = model
                self.set_status("Model loaded.")
                dpg.configure_item("run_button", enabled=True)
            except Exception as e:
                self.set_status(f"Model load failed: {e}", error=True)
            finally:
                dpg.configure_item("load_button", enabled=True, label="Reload Model")

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, sender=None, app_data=None, user_data=None):
        if self.state["running"]:
            return

        prompt     = dpg.get_value("prompt_input")
        neg_prompt = dpg.get_value("neg_prompt_input")
        steps      = int(dpg.get_value("steps_slider"))
        guidance   = dpg.get_value("guidance_slider")
        strength   = dpg.get_value("strength_slider")

        if not self.state["image_path"]:
            self.set_status("Please select an input image.", error=True); return
        if not prompt.strip():
            self.set_status("Please enter a prompt.", error=True); return
        if self.mask_np.max() == 0.0:
            self.set_status("Please draw or load a mask.", error=True); return

        self.state["running"] = True
        dpg.configure_item("run_button", enabled=False, label="Running...")
        self.set_status("Running inference...")

        def _worker():
            try:
                # Use the cropped region as input, not the full image
                cropped = self._get_cropped_image()
                image = cropped.resize((512, 512), Image.LANCZOS)
                mask  = Image.fromarray((self.mask_np * 255).astype(np.uint8), mode="L").resize((512, 512))

                pipe = build_pipeline(self.state)
                pipe.to(self.state["pipeline"].device)

                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt if neg_prompt.strip() else None,
                        image=image, mask_image=mask,
                        num_inference_steps=steps,
                        guidance_scale=guidance, strength=strength,
                    ).images[0]

                flat = self._pil_to_hwc(result).flatten()
                np.copyto(self.buf_output, flat)
                dpg.set_value("tex_output", self.buf_output)

                out_path = Path(f"{self.output_dir}/render-{time.time()}.png")
                result.save(out_path)
                self.set_status(f"Done. Saved → {out_path.resolve()}")

            except Exception as e:
                import traceback; traceback.print_exc()
                self.set_status(f"Error: {e}", error=True)
            finally:
                self.state["running"] = False
                dpg.configure_item("run_button", enabled=True, label="Run Inpainting")

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        dpg.create_context()
        self.build_ui()
        dpg.create_viewport(title="SD Inpainting", width=1700, height=980)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()