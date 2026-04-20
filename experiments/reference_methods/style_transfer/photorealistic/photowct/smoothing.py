"""
Photorealistic smoothing modules for PhotoWCT and related methods.

Provides two options:
  1. GIFSmoothing – Guided Image Filtering via OpenCV (fast, recommended).
  2. Propagator  – Laplacian matting-based propagation (no OpenCV needed,
                   but slower and requires scipy).

Both accept a stylised image and the original content image, and produce
a smoothed output that respects the content edges.
"""

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Option 1: Guided Image Filtering (requires opencv-contrib-python)
# ---------------------------------------------------------------------------

try:
    import cv2
    from cv2.ximgproc import guidedFilter as _guidedFilter

    _HAS_OPENCV_XIMGPROC = True
except ImportError:
    _HAS_OPENCV_XIMGPROC = False


class GIFSmoothing:
    """Post-processing via Guided Image Filtering (OpenCV ``ximgproc``)."""

    def __init__(self, r: int = 35, eps: float = 0.001):
        self.r = r
        self.eps = eps

    def process(self, stylised, content) -> Image.Image:
        """Apply guided filter.

        Args:
            stylised: Stylised image – PIL Image, numpy array (H,W,3 uint8),
                      or file path.
            content:  Content image – same formats.

        Returns:
            PIL Image with smoothed stylisation.
        """
        if not _HAS_OPENCV_XIMGPROC:
            raise RuntimeError(
                "GIFSmoothing requires opencv-contrib-python "
                "(cv2.ximgproc.guidedFilter). Install with: "
                "pip install opencv-contrib-python"
            )

        init_img = self._to_cv2(stylised)
        cont_img = self._to_cv2(content)

        if init_img.shape[:2] != cont_img.shape[:2]:
            cont_img = cv2.resize(
                cont_img, (init_img.shape[1], init_img.shape[0])
            )

        output = _guidedFilter(
            guide=cont_img, src=init_img, radius=self.r, eps=self.eps
        )
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output)

    @staticmethod
    def _to_cv2(img):
        if isinstance(img, str):
            return cv2.imread(img)
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        raise TypeError(f"Unsupported image type: {type(img)}")


# ---------------------------------------------------------------------------
# Option 2: Laplacian matting-based propagation (pure numpy/scipy)
# ---------------------------------------------------------------------------

try:
    import scipy.sparse
    import scipy.sparse.linalg
    from numpy.lib.stride_tricks import as_strided

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class Propagator:
    """Post-processing via closed-form Laplacian matting propagation.

    Adapted from the official NVIDIA FastPhotoStyle ``photo_smooth.py``.
    Credit to Marco Forte for the ``compute_laplacian`` implementation.
    """

    def __init__(self, beta: float = 0.9999):
        self.beta = beta

    def process(self, stylised, content) -> Image.Image:
        """Apply Laplacian propagation smoothing.

        Args:
            stylised: Stylised image – PIL Image or file path.
            content:  Content image – PIL Image or file path.

        Returns:
            PIL Image with photorealistic smoothing applied.
        """
        if not _HAS_SCIPY:
            raise RuntimeError(
                "Propagator smoothing requires scipy. "
                "Install with: pip install scipy"
            )

        if isinstance(content, str):
            content = np.asarray(Image.open(content).convert("RGB"))
        elif isinstance(content, Image.Image):
            content = np.asarray(content.convert("RGB"))
        else:
            content = np.asarray(content)

        if isinstance(stylised, str):
            B = np.asarray(Image.open(stylised).convert("RGB")).astype(np.float64) / 255
        elif isinstance(stylised, Image.Image):
            B = np.asarray(stylised.convert("RGB")).astype(np.float64) / 255
        else:
            B = np.asarray(stylised).astype(np.float64) / 255

        h1, w1, k = B.shape
        h = h1 - 4
        w = w1 - 4
        B = B[
            int((h1 - h) / 2) : int((h1 - h) / 2 + h),
            int((w1 - w) / 2) : int((w1 - w) / 2 + w),
            :,
        ]
        content_arr = np.asarray(
            Image.fromarray(content).resize((w, h), Image.BILINEAR)
        )
        B = self._replication_padding(B, 2)
        content_arr = self._replication_padding(content_arr, 2)
        content_arr = content_arr.astype(np.float64) / 255
        B = np.reshape(B, (h1 * w1, k))
        W = self._compute_laplacian(content_arr)
        W = W.tocsc()
        dd = W.sum(0)
        dd = np.sqrt(np.power(dd, -1))
        dd = dd.A.squeeze()
        D = scipy.sparse.csc_matrix(
            (dd, (np.arange(0, w1 * h1), np.arange(0, w1 * h1)))
        )
        S = D.dot(W).dot(D)
        A = scipy.sparse.identity(w1 * h1) - self.beta * S
        A = A.tocsc()
        solver = scipy.sparse.linalg.factorized(A)
        V = np.zeros((h1 * w1, k))
        V[:, 0] = solver(B[:, 0])
        V[:, 1] = solver(B[:, 1])
        V[:, 2] = solver(B[:, 2])
        V = V * (1 - self.beta)
        V = V.reshape(h1, w1, k)
        V = V[2 : 2 + h, 2 : 2 + w, :]

        return Image.fromarray(np.uint8(np.clip(V * 255.0, 0, 255.0)))

    @staticmethod
    def _compute_laplacian(img, eps=1e-7, win_rad=1):
        win_size = (win_rad * 2 + 1) ** 2
        h, w, d = img.shape
        c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
        win_diam = win_rad * 2 + 1
        indsM = np.arange(h * w).reshape((h, w))
        ravelImg = img.reshape(h * w, d)

        shape = (
            indsM.shape[0] - win_diam + 1,
            indsM.shape[1] - win_diam + 1,
            win_diam,
            win_diam,
        )
        strides = indsM.strides + indsM.strides
        win_inds = as_strided(indsM, shape=shape, strides=strides)
        win_inds = win_inds.reshape(c_h, c_w, win_size)

        winI = ravelImg[win_inds]
        win_mu = np.mean(winI, axis=2, keepdims=True)
        win_var = np.einsum("...ji,...jk ->...ik", winI, winI) / win_size - np.einsum(
            "...ji,...jk ->...ik", win_mu, win_mu
        )
        inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))
        X = np.einsum("...ij,...jk->...ik", winI - win_mu, inv)
        vals = (1 / win_size) * (
            1 + np.einsum("...ij,...kj->...ik", X, winI - win_mu)
        )
        nz_indsCol = np.tile(win_inds, win_size).ravel()
        nz_indsRow = np.repeat(win_inds, win_size).ravel()
        nz_indsVal = vals.ravel()
        L = scipy.sparse.coo_matrix(
            (nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w)
        )
        return L

    @staticmethod
    def _replication_padding(arr, pad):
        h, w, c = arr.shape
        ans = np.zeros((h + pad * 2, w + pad * 2, c))
        for i in range(c):
            ans[:, :, i] = np.pad(
                arr[:, :, i], pad_width=(pad, pad), mode="edge"
            )
        return ans
