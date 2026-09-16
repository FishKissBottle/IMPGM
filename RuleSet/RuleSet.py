import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
import heapq


class Water_RiseFall_Rule(object):
    """Generate water-mask changes from configurable rise and fall rules."""
    def __init__(self, obj_data, msk_data, dem_data, slope_data):
        super().__init__()

        if obj_data is None:
            raise ValueError("obj_data must not be None.")
        if msk_data is None:
            raise ValueError("msk_data must not be None.")
        if dem_data is None:
            raise ValueError("dem_data must not be None.")

        data = np.asarray(obj_data)
        msk = np.asarray(msk_data)
        if msk.ndim != 2:
            raise ValueError(f"msk_data must be a two-dimensional array, got shape {msk.shape}.")
        if msk.size == 0:
            raise ValueError("msk_data must not be empty.")
        if not np.all(np.isfinite(msk)) or not np.all(np.isin(msk, (0, 1))):
            raise ValueError("msk_data must be a finite binary mask containing only 0 and 1.")
        if not np.any(msk == 1):
            raise ValueError("msk_data must contain at least one water pixel with value 1.")
        if data.ndim < 2 or tuple(data.shape[-2:]) != tuple(msk.shape):
            raise ValueError(
                "obj_data and msk_data must have matching spatial dimensions, "
                f"got {data.shape} and {msk.shape}."
            )
        dem_data = np.asarray(dem_data)
        if dem_data.ndim != 2 or dem_data.shape != msk.shape:
            raise ValueError(
                f"dem_data must have shape {msk.shape}, got {dem_data.shape}."
            )
        if slope_data is not None:
            slope_data = np.asarray(slope_data)
            if slope_data.ndim != 2 or slope_data.shape != msk.shape:
                raise ValueError(
                    f"slope_data must have shape {msk.shape}, got {slope_data.shape}."
                )

        dem_valid = np.isfinite(dem_data)
        if not np.any(dem_valid):
            raise ValueError("dem_data must contain at least one finite elevation value.")

        # Normalized convolution prevents invalid DEM cells from acting as zero-elevation terrain.
        dem_values = np.where(dem_valid, dem_data, 0.0).astype(np.float32)
        dem_weights = gaussian_filter(dem_valid.astype(np.float32), sigma=7)
        dem = gaussian_filter(dem_values, sigma=7) / np.maximum(dem_weights, 1e-6)
        dem[~dem_valid] = np.nan

        slope = None
        if slope_data is not None:
            slope = slope_data
            slope = np.nan_to_num(slope, nan=0.0)
            slope = gaussian_filter(slope, sigma=7)

        # If no slope is provided
        if slope is None:
            dem_valid = np.isfinite(dem)
            if not np.any(dem_valid):
                raise ValueError("A slope cannot be derived because the DEM has no finite values.")
            if np.all(dem_valid):
                dem_for_slope = dem.astype(np.float64)
            else:
                _, nearest_valid_indices = distance_transform_edt(
                    ~dem_valid,
                    return_indices=True,
                )
                dem_for_slope = dem[tuple(nearest_valid_indices)].astype(np.float64)

            cellsize = 10
            random_altitude = np.random.randint(50, 2000)
            dzdx = ndimage.sobel(dem_for_slope * random_altitude, axis=1) / (8 * cellsize)
            dzdy = ndimage.sobel(dem_for_slope * random_altitude, axis=0) / (8 * cellsize)
            slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

            slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

            slope = (slope - np.min(slope)) / (np.max(slope) - np.min(slope))
            slope[~dem_valid] = np.nan

        self.data = data
        self.msk = msk
        self.dem = dem.astype(np.float32)
        self.slope = slope.astype(np.float32)

        self.valid = np.isfinite(self.dem)
        self.water_bool = (self.msk > 0) & self.valid
        if not np.any(self.water_bool):
            raise ValueError("No water pixels overlap valid DEM cells.")

        self.hand = None


    @staticmethod
    def _normalize_0_1(x, clip_percentiles=(2, 98)):
        x = x.astype(np.float32)
        lo, hi = np.nanpercentile(x, clip_percentiles)
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-6)
    

    def _neighbors(self, r, c, h, w, connectivity=8):
        if connectivity == 4:
            dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        else:
            dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in dirs:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


    def _is_boundary(self, flooded: np.ndarray, r: int, c: int, connectivity: int = 8) -> bool:
        h, w = flooded.shape
        for rr, cc in self._neighbors(r, c, h, w, connectivity):
            if not flooded[rr, cc]:
                return True
        return False


    def compute_hand(self):

        if not np.any(self.msk):
            raise ValueError("No water pixels found in water mask (water>0).")

        # distance_transform_edt: compute distance for True pixels to nearest False pixel
        # We want distance from non-water to water => input should be ~water_bool
        dist, (iy, ix) = distance_transform_edt(~self.water_bool, return_indices=True)

        nearest_water_dem = self.dem[iy, ix]
        hand = self.dem - nearest_water_dem
        hand[self.water_bool] = 0.0

        # invalid stays nan
        hand[~self.valid] = np.nan

        self.dist_to_water = dist.astype(np.float32)
        self.nearest_water_dem = nearest_water_dem.astype(np.float32)
        self.hand = hand.astype(np.float32)
        return self.hand


    def susceptibility_score(self, w_hand=0.60, w_dist: float = 0.30, w_slope: float = 0.10, clip_percentiles=(2, 98)):

        weights = np.asarray((w_hand, w_dist, w_slope), dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("w_hand, w_dist, and w_slope must be finite non-negative values.")
        if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "w_hand, w_dist, and w_slope must sum to 1.0, "
                f"got {float(weights.sum()):.8f}."
            )
        if len(clip_percentiles) != 2:
            raise ValueError("clip_percentiles must contain exactly two values.")
        clip_lo, clip_hi = clip_percentiles
        if not (0 <= clip_lo < clip_hi <= 100):
            raise ValueError(
                "clip_percentiles must satisfy 0 <= low < high <= 100, "
                f"got {clip_percentiles}."
            )

        if self.hand is None or self.dist_to_water is None:
            self.compute_hand()

        hand_n = 1.0 - self._normalize_0_1(self.hand, clip_percentiles=clip_percentiles)
        dist_n = 1.0 - self._normalize_0_1(self.dist_to_water, clip_percentiles=clip_percentiles)
        slope_n = 1.0 - self._normalize_0_1(self.slope, clip_percentiles=clip_percentiles)

        score = (w_hand * hand_n) + (w_dist * dist_n) + (w_slope * slope_n)
        score[~self.valid] = np.nan

        score = np.clip(score, 0.0, 1.0).astype(np.float32)
        return score


    def grow_inundation(self, delta_num, delta_rate, score, connectivity=8):

        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be either 4 or 8, got {connectivity}.")
        if score is None:
            raise ValueError("score must not be None.")
        score = np.asarray(score)
        if score.ndim != 2 or score.shape != self.msk.shape:
            raise ValueError(f"score must have shape {self.msk.shape}, got {score.shape}.")
        if delta_num is not None:
            if not np.isscalar(delta_num) or not np.isfinite(delta_num) or delta_num < 0:
                raise ValueError("delta_num must be a finite non-negative integer or None.")
            if not float(delta_num).is_integer():
                raise ValueError(f"delta_num must be an integer, got {delta_num}.")
        if delta_rate is not None:
            if not np.isscalar(delta_rate) or not np.isfinite(delta_rate) or delta_rate < 0:
                raise ValueError("delta_rate must be a finite non-negative value or None.")

        flooded = self.msk.astype(bool).copy()

        valid = np.isfinite(self.dem)
        flooded &= valid

        candidate = valid.copy()
        candidate &= ~flooded

        if delta_num is None and delta_rate is None:
            raise ValueError("Set either delta_num or delta_rate.")
        if delta_num is None:
            water_pixel_count = np.count_nonzero(self.msk == 1)
            delta_num = delta_rate * water_pixel_count

        target_pixels = int(delta_num)
        if target_pixels <= 0:
            out = flooded.astype(np.uint8)
            return out, np.zeros_like(out, dtype=np.uint8)

        # Preprocess score: NaN does not participate (treated as -inf)
        score_work = score.astype(np.float32).copy()
        score_work[~np.isfinite(score_work)] = -np.inf

        # Build a queue of (-score, r, c): higher scores expand first
        h, w = flooded.shape
        heap = []
        in_heap = np.zeros_like(flooded, dtype=bool)
        delta_wet = np.zeros_like(flooded, dtype=np.uint8)

        # Initialize: push neighbors of the flooded area into the heap as the frontier (highest score first)
        flooded_idx = np.argwhere(flooded)
        for r, c in flooded_idx:
            for rr, cc in self._neighbors(r, c, h, w, connectivity):
                if candidate[rr, cc] and (not in_heap[rr, cc]):
                    sc = score_work[rr, cc]
                    if sc > -np.inf:
                        heapq.heappush(heap, (-float(sc), rr, cc))
                        in_heap[rr, cc] = True

        added = 0
        while heap and added < target_pixels:
            neg_sc, r, c = heapq.heappop(heap)

            if flooded[r, c] or (not candidate[r, c]):
                continue

            flooded[r, c] = True
            candidate[r, c] = False
            delta_wet[r, c] = 1
            added += 1

            # Neighbors of the newly added cell enter the frontier
            for rr, cc in self._neighbors(r, c, h, w, connectivity):
                if candidate[rr, cc] and (not in_heap[rr, cc]):
                    sc = score_work[rr, cc]
                    if sc > -np.inf:
                        heapq.heappush(heap, (-float(sc), rr, cc))
                        in_heap[rr, cc] = True

        return flooded.astype(np.float32), delta_wet



    def shrink_inundation(self, delta_num, delta_rate, score, connectivity = 8):

        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be either 4 or 8, got {connectivity}.")
        if score is None:
            raise ValueError("score must not be None.")
        score = np.asarray(score)
        if score.ndim != 2 or score.shape != self.msk.shape:
            raise ValueError(f"score must have shape {self.msk.shape}, got {score.shape}.")
        if delta_num is not None:
            if not np.isscalar(delta_num) or not np.isfinite(delta_num) or delta_num < 0:
                raise ValueError("delta_num must be a finite non-negative integer or None.")
            if not float(delta_num).is_integer():
                raise ValueError(f"delta_num must be an integer, got {delta_num}.")
        if delta_rate is not None:
            if not np.isscalar(delta_rate) or not np.isfinite(delta_rate) or delta_rate < 0:
                raise ValueError("delta_rate must be a finite non-negative value or None.")

        flooded = self.msk.astype(bool).copy()
        valid = np.isfinite(self.dem)
        flooded &= valid

        if delta_num is None and delta_rate is None:
            raise ValueError("Set either delta_num or delta_rate.")
        if delta_num is None:
            water_pixel_count = np.count_nonzero(self.msk == 1)
            delta_num = delta_rate * water_pixel_count

        target_pixels = int(delta_num)
        if target_pixels <= 0:
            out = flooded.astype(np.uint8)
            return out, np.zeros_like(out, dtype=np.uint8)
        
        # Preprocess score: NaN is not removed first (treated as +inf)
        score_work = score.astype(np.float32).copy()
        score_work[~np.isfinite(score_work)] = np.inf

        h, w = flooded.shape
        heap = []
        delta_dry = np.zeros_like(flooded, dtype=np.uint8)

        # Initialize: push all removable boundary cells into the heap (lowest score first)
        flooded_idx = np.argwhere(flooded)
        for r, c in flooded_idx:
            if self._is_boundary(flooded, r, c, connectivity):
                heapq.heappush(heap, (float(score_work[r, c]), r, c))

        removed = 0
        while heap and removed < target_pixels:
            sc, r, c = heapq.heappop(heap)

            # Re-check after state changes
            if not flooded[r, c]:
                continue
            if not self._is_boundary(flooded, r, c, connectivity):
                continue

            flooded[r, c] = False
            delta_dry[r, c] = 1
            removed += 1

            # After removal: its neighbors may become new boundary cells, push them into the heap
            for rr, cc in self._neighbors(r, c, h, w, connectivity):
                if flooded[rr, cc]:
                    # No need to deduplicate: duplicates are allowed, boundary state is re-checked on pop
                    heapq.heappush(heap, (float(score_work[rr, cc]), rr, cc))

        return flooded.astype(np.float32), delta_dry


        
if __name__=='__main__':

    import matplotlib.pyplot as plt
    from IMPGM_TifReader import Tif_Read_and_Write

    h_start_idx, h_end_idx =   0,  448
    w_start_idx, w_end_idx = 900, 1348

    # Read FgGen_cond_ele
    Msk_data, _, _ = Tif_Read_and_Write().Tif_Read(r'./data/large_image/sentinel12_s2_10_mask.tif')
    Msk_data = Msk_data[h_start_idx: h_end_idx, w_start_idx: w_end_idx]

    # Read ImgSyn_cond_ele
    Img_data, Img_projection, Img_geotransform = Tif_Read_and_Write().Tif_Read(r'./data/large_image/sentinel12_s2_10_data.tif')
    Img_data = Img_data[:, h_start_idx: h_end_idx, w_start_idx: w_end_idx]

    DEM_data, _, _ = Tif_Read_and_Write().Tif_Read(r'./data/large_image/sentinel12_s2_10_dem.tif')
    DEM_data = DEM_data[h_start_idx: h_end_idx, w_start_idx: w_end_idx]
    Slope_data, _, _ = Tif_Read_and_Write().Tif_Read(r'./data/large_image/sentinel12_s2_10_slope.tif')
    Slope_data = Slope_data[h_start_idx: h_end_idx, w_start_idx: w_end_idx]

    delta_num = None
    delta_rate = 0.4

    rule = Water_RiseFall_Rule(obj_data=Img_data, msk_data=Msk_data, dem_data=DEM_data, slope_data=Slope_data)
    score_map = rule.susceptibility_score()
    flooded1, delta_mask = rule.grow_inundation(delta_num=delta_num, delta_rate=delta_rate, score=score_map)



    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(DEM_data, cmap='gray', interpolation='nearest')    
    plt.subplot(2, 2, 2)
    plt.imshow(Slope_data, cmap='gray', interpolation='nearest')
    plt.subplot(2, 2, 3)
    plt.imshow(score_map, cmap='gray', interpolation='nearest')
    plt.subplot(2, 2, 4)
    plt.imshow(flooded1, cmap='gray', interpolation='nearest')
    plt.show()

