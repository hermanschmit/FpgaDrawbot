"""
Guilloche-style continuous line drawing, in the spirit of banknote engraving.

The whole drawing is a single Archimedean spiral filling an ellipse (or
circle) inscribed in the image, so the pen never lifts and there is no seam
where concentric rings would otherwise have to be joined. Three things are
layered on top of the bare spiral, all expressed in units of the spiral
pitch (the radial distance between adjacent turns):

- rosette: a decorative radial modulation r += rosette_amp*sin(lobes*theta +
  twist*theta). With twist=0.5 each turn is shifted half a lobe against the
  previous one, and once rosette_amp exceeds 0.5 adjacent turns cross, giving
  the classic woven guilloche lattice. Purely ornamental, independent of the
  image.
- wave (the tone carrier): a fine sinusoid along the curve's normal whose
  amplitude is proportional to local darkness, and whose frequency rises with
  darkness by a factor of (1 + freq_mod). More ink per unit area where the
  image is dark, a plain line where it is white. The phase is accumulated
  along arc length so the wavelength stays constant in pixels regardless of
  radius.
- relief: a radial displacement proportional to brightness, which gives the
  embossed "line displacement" look of engraved portraits. Off by default.

Coordinates follow the repo convention: points are (row, col), which
Segments.binWrite() emits as (col, row).
"""
import numpy as np
from scipy import ndimage
from skimage.transform import resize

import Segments


class Guilloche:
    def __init__(self, im, canvas_size=800, turns=80, shape='ellipse',
                 wavelength=1.0, amplitude=0.9, freq_mod=1.0,
                 rosette_amp=0.0, rosette_lobes=24, rosette_twist=0.5,
                 relief=0.0, gamma=0.7, white_cut=0.1, pts_per_wave=6,
                 white_step=1.5):
        """
        im: grayscale float image, 0 = black, 255 = white.
        canvas_size: the long image side is resized to this many pixels.
        turns: number of spiral turns from center to edge (line density).
        shape: 'ellipse' fills the image aspect ratio, 'circle' is a medallion.
        wavelength, amplitude: tone-wave wavelength at white and peak
            amplitude at full black, as fractions of the pitch. amplitude
            above 0.5 lets dark turns overlap their neighbors.
        freq_mod: wave frequency at full black is (1 + freq_mod) times that
            at white.
        rosette_amp, rosette_lobes, rosette_twist: see module docstring;
            rosette_amp is in pitches, rosette_twist in lobes per turn.
        relief: peak radial displacement at full white, in pitches.
        gamma: applied to darkness (1 - im/255); <1 darkens midtones.
        white_cut: after auto-levels (2nd..98th percentile stretched to
            0..1), darkness below this is treated as paper white, so light
            backgrounds get a plain line instead of a faint wave.
        pts_per_wave: samples per tone-wave period where the wave is active.
        white_step: point spacing (pixels) kept where the wave is inactive.
        """
        self.segments = Segments.Segments()

        im = np.asarray(im, dtype=float)
        h0, w0 = im.shape
        s = canvas_size / max(h0, w0)
        h, w = max(2, int(round(h0 * s))), max(2, int(round(w0 * s)))
        dark = np.clip(1.0 - resize(im, (h, w), anti_aliasing=True) / 255.0, 0.0, 1.0)
        lo, hi = np.percentile(dark, [2, 98])
        dark = np.clip((dark - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        dark = np.clip((dark - white_cut) / (1.0 - white_cut), 0.0, 1.0) ** gamma

        # Radii of the inscribed shape, shrunk so every modulation stays on canvas.
        r_row, r_col = (h - 1) / 2.0, (w - 1) / 2.0
        if shape == 'circle':
            r_row = r_col = min(r_row, r_col)
        elif shape != 'ellipse':
            raise ValueError("shape must be 'ellipse' or 'circle'")
        overshoot = rosette_amp + amplitude + relief
        # pitch = r_min / turns, with r_min = min radius - overshoot * pitch
        pitch = min(r_row, r_col) / (turns + overshoot)
        self.pitch = pitch
        r_min = pitch * turns
        stretch_row = (r_row - overshoot * pitch) / r_min
        stretch_col = (r_col - overshoot * pitch) / r_min
        c_row, c_col = (h - 1) / 2.0, (w - 1) / 2.0

        # Smooth the tone map on the scale of one pitch so the wave amplitude
        # varies smoothly along the line.
        dark = ndimage.gaussian_filter(dark, sigma=max(0.5, 0.5 * pitch))

        # Sample the spiral roughly uniformly in arc length. For r = b*theta,
        # s ~= b*theta^2/2; b is taken from the larger radius so the step is
        # small enough everywhere on the ellipse.
        theta_max = 2.0 * np.pi * turns
        b = r_min * max(stretch_row, stretch_col) / theta_max
        # Oversample here (4x); decimation below brings it back down.
        ds = pitch * wavelength / ((1.0 + freq_mod) * pts_per_wave * 4)
        s_total = b * theta_max ** 2 / 2.0
        theta = np.sqrt(2.0 * np.arange(0.0, s_total + ds, ds) / b)
        theta = np.minimum(theta, theta_max)

        # Base radius in pixels (circle space), plus rosette.
        radius = r_min * theta / theta_max
        if rosette_amp > 0:
            # Fade in over the inner 20% of the radius: near the center the
            # lobes get shorter than the pitch and collapse into a starburst.
            fade = np.minimum(1.0, theta / (0.2 * theta_max))
            radius = radius + fade * rosette_amp * pitch * np.sin((rosette_lobes + rosette_twist) * theta)

        def place(rad):
            return np.column_stack((c_row - stretch_row * rad * np.sin(theta),
                                    c_col + stretch_col * rad * np.cos(theta)))

        pts = place(radius)
        d = ndimage.map_coordinates(dark, pts.T, order=1, mode='nearest')
        if relief > 0:
            pts = place(radius + relief * pitch * (1.0 - d))

        # Unit normals and arc length of the base curve.
        tan = np.gradient(pts, axis=0)
        tlen = np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-12)
        normal = np.column_stack((-tan[:, 1], tan[:, 0])) / tlen[:, None]
        seg = np.concatenate(([0.0], np.hypot(*np.diff(pts, axis=0).T)))

        freq = (1.0 + freq_mod * d) / (wavelength * pitch)
        phase = 2.0 * np.pi * np.cumsum(freq * seg)
        amp = amplitude * pitch * d
        pts = pts + normal * (amp * np.sin(phase))[:, None]

        # Decimate: where the wave is active keep pts_per_wave samples per
        # period (wherever the phase crosses a multiple of 2*pi/pts_per_wave,
        # so low-frequency stretches aren't oversampled); elsewhere keep one
        # sample every white_step pixels. A one-sample margin around active
        # stretches keeps wave onsets intact.
        active = amp > 0.02 * pitch
        active = np.convolve(active, np.ones(3), mode='same') > 0
        tick = np.floor(phase * pts_per_wave / (2.0 * np.pi))
        on_tick = np.concatenate(([True], np.diff(tick) > 0))
        k = max(1, int(white_step / ds))
        keep = (active & on_tick) | (np.arange(len(pts)) % k == 0)
        keep[0] = True
        keep[-1] = True
        pts = pts[keep]

        self.segments.append(pts)
        print("Guilloche: pitch %.2f px, %d points" % (pitch, len(pts)))

    def binWrite(self, fn):
        self.segments.binWrite(fn)
