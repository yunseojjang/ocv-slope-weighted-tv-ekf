Slope-TV EKF 기반 배터리 SOC 추정 코드
본 코드는 학위논문 방법론에 맞추어 다음 절차로 구성하였다.

1. Data Preparation
   - DST 전류-전압 데이터와 Reference SOC 데이터를 불러온다.
   - 각 셀의 OCV-SOC LUT를 불러온다.
   - 평가 구간은 100% SOC에서 20% SOC까지로 제한한다.

2. Independent Filter Tuning
   - Test data를 사용하지 않고 별도의 tuning data에서만 필터 파라미터를 선정한다.
   - Wavelet, Low-pass, Savitzky-Golay, Fixed-TV, Slope-TV의 후보 파라미터를 비교한다.

3. Common 2-RC Model
   - LFP와 NCA 모두 동일한 2-RC ECM 구조를 사용한다.
   - 단, OCV-SOC LUT는 화학계별/셀별 특성에 맞게 별도로 사용한다.

4. Voltage Preprocessing
   - Raw voltage
   - Wavelet denoising
   - Low-pass filtering
   - Savitzky-Golay filtering
   - Fixed-TV denoising
   - OCV-slope-weighted TV denoising

5. EKF-Based SOC Estimation
   - 상태변수: x = [SOC, V1, V2]^T
   - V1, V2는 각각 2-RC 모델의 polarization voltage이다.
   - 전압 측정값을 이용해 EKF correction을 수행한다.

6. Performance Evaluation
   - RMSE, MAE, MAXE를 SOC [%] 단위로 계산한다.
   - 필터별 SOC 추정 성능을 비교한다.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, savgol_filter
import pywt

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:
    denoise_tv_chambolle = None
    print("[WARN] scikit-image가 설치되지 않았습니다. Fixed-TV와 Slope-TV를 사용하려면 설치가 필요합니다.")
    print("       pip install scikit-image")


# 1. 기본 설정

DT_DEFAULT_SEC = 1.0

# 전류 부호 정의
# 본 코드에서는 I > 0을 방전 전류로 정의한다.
# 만약 측정 장비에서 방전 전류가 음수로 저장되면 FLIP_CURRENT_SIGN = True로 설정한다.
FLIP_CURRENT_SIGN = True

# EKF 노이즈 공분산
# SOC와 RC 전압 상태의 process noise
Q_PROCESS = np.diag([1e-8, 1e-6, 1e-6])

# 전압 측정 noise variance
R_MEAS = 2.5e-4

# 평가 구간
SOC_START_TARGET = 1.00
SOC_END_TARGET = 0.20

# 2. 데이터 구조 정의

@dataclass
class ModelParams2RC:
    """
    2-RC ECM 파라미터

    R0 : Ohmic resistance
    R1, C1 : 첫 번째 RC branch
    R2, C2 : 두 번째 RC branch
    """
    R0: float
    R1: float
    C1: float
    R2: float
    C2: float


@dataclass
class BatteryRecord:
    """
    하나의 배터리 셀 데이터 묶음

    dataset     : 예) 4Ah LFP, 6Ah LFP, 3.4Ah NCA
    cell_id     : 셀 번호
    chemistry   : LFP 또는 NCA
    Qn_Ah       : 정격 용량 [Ah]
    time_s      : 시간 [s]
    current_A   : 전류 [A], 방전 방향을 양수로 통일
    voltage_V   : 단자전압 [V]
    soc_true    : Reference SOC [0~1]
    ocv_lut     : OCV-SOC 보간 함수
    """
    dataset: str
    cell_id: int
    chemistry: str
    Qn_Ah: float
    time_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray
    soc_true: np.ndarray
    ocv_lut: object

# 3. OCV-SOC LUT


class OCVLUT:
    """
    OCV-SOC lookup table

    EKF에서 전압 예측식은 다음과 같다.

        Vt = OCV(SOC) - V1 - V2 - R0*I

    따라서 EKF correction을 위해 OCV(SOC)뿐만 아니라
    dOCV/dSOC도 필요하다.
    """

    def __init__(self, soc_grid, ocv_grid):
        soc = np.asarray(soc_grid, dtype=float)
        ocv = np.asarray(ocv_grid, dtype=float)

        # SOC가 0~100 [%]로 들어온 경우 0~1로 변환
        if np.nanmax(soc) > 1.5:
            soc = soc / 100.0

        mask = np.isfinite(soc) & np.isfinite(ocv)
        soc = soc[mask]
        ocv = ocv[mask]

        idx = np.argsort(soc)
        self.soc = soc[idx]
        self.ocv = ocv[idx]

        # 중복 SOC 제거
        self.soc, unique_idx = np.unique(self.soc, return_index=True)
        self.ocv = self.ocv[unique_idx]

        # dOCV/dSOC 계산
        self.docv_dsoc = np.gradient(self.ocv, self.soc)

    def ocv_value(self, soc):
        soc = np.clip(soc, 0.0, 1.0)
        return np.interp(soc, self.soc, self.ocv)

    def slope_value(self, soc):
        soc = np.clip(soc, 0.0, 1.0)
        return np.interp(soc, self.soc, self.docv_dsoc)

    def slope_reference(self):
        """
        Slope-TV에서 기준 기울기로 사용할 대표값.
        OCV plateau 구간에서 기울기가 매우 작아지는 문제를 완화하기 위해
        전체 OCV slope의 median 값을 사용한다.
        """
        slope_abs = np.abs(self.docv_dsoc)
        slope_abs = slope_abs[np.isfinite(slope_abs)]
        slope_abs = slope_abs[slope_abs > 1e-8]

        if len(slope_abs) == 0:
            return 1.0

        return float(np.median(slope_abs))

# 4. 엑셀 데이터 읽기 함수

def find_column(df, keywords):
    """
    엑셀 파일마다 column 이름이 조금씩 다를 수 있으므로,
    keyword를 포함하는 column을 자동으로 찾는다.
    """
    normalized = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    for key in keywords:
        key = key.lower().replace(" ", "").replace("_", "")
        for norm_col, original_col in normalized.items():
            if key in norm_col:
                return original_col

    raise ValueError(f"다음 키워드에 해당하는 column을 찾지 못했습니다: {keywords}")


def read_ocv_table(ocv_path):
    """
    OCV-SOC 파일을 읽고 OCVLUT 객체로 변환한다.
    """
    df = pd.read_excel(ocv_path)

    soc_col = find_column(df, ["soc"])
    ocv_col = find_column(df, ["ocv", "voltage", "volt"])

    soc = df[soc_col].to_numpy(dtype=float)
    ocv = df[ocv_col].to_numpy(dtype=float)

    return OCVLUT(soc, ocv)


def read_dst_data(dst_path, dt_default=1.0):
    """
    DST 전류-전압 데이터를 읽는다.

    필요한 column:
    - time 또는 step time
    - current
    - voltage
    - soc
    """
    df = pd.read_excel(dst_path)

    current_col = find_column(df, ["current", "curr"])
    voltage_col = find_column(df, ["voltage", "volt"])
    soc_col = find_column(df, ["soc"])

    current = df[current_col].to_numpy(dtype=float)
    voltage = df[voltage_col].to_numpy(dtype=float)
    soc = df[soc_col].to_numpy(dtype=float)

    # SOC가 [%] 단위이면 0~1로 변환
    if np.nanmax(soc) > 1.5:
        soc = soc / 100.0

    # current가 mA 단위로 저장된 경우 A로 변환
    if np.nanmax(np.abs(current)) > 100:
        current = current / 1000.0

    # 장비 전류 부호가 방전 음수이면 부호 반전
    if FLIP_CURRENT_SIGN:
        current = -current

    # time column이 있으면 사용하고, 없으면 dt_default로 생성
    try:
        time_col = find_column(df, ["time"])
        time_s = df[time_col].to_numpy(dtype=float)
        dt = np.diff(time_s, prepend=time_s[0])
        dt[dt <= 0] = dt_default
    except ValueError:
        dt = np.ones_like(voltage) * dt_default
        time_s = np.cumsum(dt) - dt[0]

    mask = (
        np.isfinite(time_s)
        & np.isfinite(current)
        & np.isfinite(voltage)
        & np.isfinite(soc)
        & np.isfinite(dt)
    )

    return time_s[mask], current[mask], voltage[mask], soc[mask], dt[mask]


def cut_soc_window(time_s, current_A, voltage_V, soc_true, dt,
                   start_target=1.0, end_target=0.2):
    """
    논문 평가 조건에 맞게 100% SOC에서 20% SOC 구간만 사용한다.
    """
    soc_true = np.asarray(soc_true, dtype=float)

    start_candidates = np.where(soc_true >= min(start_target, 0.99))[0]
    if len(start_candidates) > 0:
        start_idx = int(start_candidates[0])
    else:
        start_idx = int(np.nanargmax(soc_true))

    end_candidates = np.where(soc_true[start_idx:] <= end_target)[0]
    if len(end_candidates) > 0:
        end_idx = start_idx + int(end_candidates[0])
    else:
        end_idx = len(soc_true) - 1

    sl = slice(start_idx, end_idx + 1)

    return (
        time_s[sl] - time_s[sl][0],
        current_A[sl],
        voltage_V[sl],
        soc_true[sl],
        dt[sl],
    )

# 5. Voltage preprocessing filters

def wavelet_denoising(v, wavelet="db4", level=2):
    """
    Wavelet denoising

    전압 신호를 wavelet domain으로 변환한 후,
    detail coefficient에 soft-thresholding을 적용하여 고주파 잡음을 제거한다.
    """
    v = np.asarray(v, dtype=float)

    if len(v) < 8:
        return v.copy()

    max_level = pywt.dwt_max_level(len(v), pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(int(level), max_level))

    coeff = pywt.wavedec(v, wavelet, mode="per", level=level)

    sigma = np.median(np.abs(coeff[-1] - np.median(coeff[-1]))) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(len(v)))

    coeff[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeff[1:]]
    v_filtered = pywt.waverec(coeff, wavelet, mode="per")

    return v_filtered[:len(v)]


def lowpass_filter(v, cutoff=0.05, fs=1.0, order=3):
    """
    Low-pass filter

    cutoff 이하의 저주파 성분은 통과시키고,
    고주파 노이즈는 감쇠시킨다.
    """
    v = np.asarray(v, dtype=float)

    if len(v) < order * 6:
        return v.copy()

    nyq = 0.5 * fs
    wc = np.clip(cutoff / nyq, 1e-6, 0.999999)

    b, a = butter(order, wc, btype="low")
    return filtfilt(b, a, v)


def savgol_denoising(v, window_length=31, polyorder=3):
    """
    Savitzky-Golay filter

    일정 window 내에서 다항식 fitting을 수행하여 신호를 평활화한다.
    전압 곡선 형태 보존에는 유리하지만, window가 길면 과도 응답이 완만해질 수 있다.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)

    window_length = int(window_length)

    if window_length % 2 == 0:
        window_length += 1

    if window_length >= n:
        window_length = n - 1 if n % 2 == 0 else n

    if window_length <= polyorder + 2:
        return v.copy()

    return savgol_filter(v, window_length=window_length, polyorder=polyorder)


def fixed_tv_denoising(v, weight=0.1):
    """
    Fixed-TV denoising

    전체 SOC 구간에서 동일한 TV weight를 적용한다.
    급격한 전압 변화는 보존하면서 작은 고주파 변동을 억제하는 특징이 있다.
    """
    v = np.asarray(v, dtype=float)

    if denoise_tv_chambolle is None:
        return v.copy()

    return denoise_tv_chambolle(v, weight=float(weight))


def apply_voltage_filter(v_raw, method, param):
    """
    필터 방법에 따라 전압 입력을 생성한다.
    """
    if method == "raw":
        return np.asarray(v_raw, dtype=float)

    if method == "wavelet":
        return wavelet_denoising(v_raw, level=param["level"])

    if method == "lowpass":
        return lowpass_filter(v_raw, cutoff=param["cutoff"], fs=param.get("fs", 1.0))

    if method == "savgol":
        return savgol_denoising(v_raw, window_length=param["window_length"])

    if method == "fixed_tv":
        return fixed_tv_denoising(v_raw, weight=param["weight"])

    raise ValueError(f"정의되지 않은 필터 방법입니다: {method}")

# 6. Slope-TV weight 계산

def compute_slope_tv_weight(base_weight, dOCV_dSOC, ocv_lut,
                            weight_min=0.005, weight_max=0.30,
                            factor_min=0.60, factor_max=3.00,
                            eps=1e-6):
    """
    OCV slope 기반 Slope-TV weight 계산

    핵심 아이디어:
    - OCV-SOC 곡선의 기울기가 작은 plateau 구간에서는
      전압 변화가 SOC 변화에 둔감하므로 measurement noise가 SOC correction에 더 큰 영향을 줄 수 있다.
    - 따라서 |dOCV/dSOC|가 작을수록 TV weight를 증가시켜 전압 노이즈를 더 강하게 억제한다.
    - 반대로 |dOCV/dSOC|가 큰 구간에서는 과도한 smoothing을 방지하기 위해 weight를 낮춘다.

    lambda_k = base_weight * clip(slope_ref / (|dOCV/dSOC| + eps), factor_min, factor_max)
    """
    slope_abs = max(abs(float(dOCV_dSOC)), eps)
    slope_ref = max(float(ocv_lut.slope_reference()), eps)

    factor = slope_ref / slope_abs
    factor = np.clip(factor, factor_min, factor_max)

    weight = base_weight * factor
    weight = np.clip(weight, weight_min, weight_max)

    return float(weight)


def causal_slope_tv_last_value(v_raw, k, weight_k, window=41):
    """
    Causal sliding-window Slope-TV denoising

    현재 시점 k의 전압을 보정할 때 미래 데이터를 사용하지 않기 위해
    v[0:k] 또는 최근 window 구간만 사용한다.

    이 함수는 denoised window의 마지막 값만 EKF update에 사용한다.
    """
    if denoise_tv_chambolle is None:
        return float(v_raw[k])

    start = max(0, k - window + 1)
    v_window = np.asarray(v_raw[start:k + 1], dtype=float)

    if len(v_window) < 4:
        return float(v_raw[k])

    v_tv = denoise_tv_chambolle(v_window, weight=float(weight_k))
    return float(v_tv[-1])


# 7. 2-RC EKF SOC estimation

def ekf_2rc_soc_estimation(current_A,
                           voltage_V,
                           dt,
                           ocv_lut,
                           Qn_Ah,
                           params: ModelParams2RC,
                           soc0=1.0,
                           eta=1.0,
                           slope_tv=False,
                           slope_tv_base_weight=0.1,
                           slope_tv_window=41):
    """
    2-RC ECM 기반 EKF SOC 추정

    상태변수:
        x = [SOC, V1, V2]^T

    상태방정식:
        SOC_k+1 = SOC_k - eta * I_k * dt / Qn
        V1_k+1  = exp(-dt/(R1*C1))*V1_k + (1-exp(-dt/(R1*C1)))*R1*I_k
        V2_k+1  = exp(-dt/(R2*C2))*V2_k + (1-exp(-dt/(R2*C2)))*R2*I_k

    출력방정식:
        Vt_k = OCV(SOC_k) - V1_k - V2_k - R0*I_k

    여기서 I > 0은 방전 전류이다.
    """
    n = len(voltage_V)

    # 초기 상태
    x = np.array([soc0, 0.0, 0.0], dtype=float)

    # 초기 오차 공분산
    P = np.diag([1e-4, 1e-3, 1e-3])
    I3 = np.eye(3)

    Qn_As = Qn_Ah * 3600.0

    soc_hat = np.zeros(n)
    voltage_hat = np.zeros(n)
    voltage_used = np.zeros(n)
    slope_tv_weight_hist = np.full(n, np.nan)

    for k in range(n):
        Ik = float(current_A[k])
        dtk = float(dt[k])

        # -----------------------------
        # Prediction step
        # -----------------------------
        soc_pred = x[0] - eta * Ik * dtk / Qn_As
        soc_pred = np.clip(soc_pred, 0.0, 1.0)

        a1 = np.exp(-dtk / (params.R1 * params.C1))
        a2 = np.exp(-dtk / (params.R2 * params.C2))

        v1_pred = a1 * x[1] + (1.0 - a1) * params.R1 * Ik
        v2_pred = a2 * x[2] + (1.0 - a2) * params.R2 * Ik

        x_pred = np.array([soc_pred, v1_pred, v2_pred], dtype=float)

        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, a1,  0.0],
            [0.0, 0.0,  a2],
        ])

        P_pred = F @ P @ F.T + Q_PROCESS

        # 예측 전압
        ocv_pred = ocv_lut.ocv_value(x_pred[0])
        v_pred = ocv_pred - x_pred[1] - x_pred[2] - params.R0 * Ik

        # OCV slope
        dOCV = ocv_lut.slope_value(x_pred[0])

        # -----------------------------
        # Slope-TV preprocessing
        # -----------------------------
        if slope_tv:
            weight_k = compute_slope_tv_weight(
                base_weight=slope_tv_base_weight,
                dOCV_dSOC=dOCV,
                ocv_lut=ocv_lut,
            )

            vk = causal_slope_tv_last_value(
                voltage_V,
                k,
                weight_k=weight_k,
                window=slope_tv_window,
            )

            slope_tv_weight_hist[k] = weight_k

        else:
            vk = float(voltage_V[k])

        voltage_used[k] = vk

        # -----------------------------
        # Correction step
        # -----------------------------
        H = np.array([[dOCV, -1.0, -1.0]], dtype=float)

        residual = vk - v_pred
        S = H @ P_pred @ H.T + R_MEAS
        S = max(float(S[0, 0]), 1e-12)

        K = (P_pred @ H.T) / S

        x = x_pred + K.flatten() * residual
        x[0] = np.clip(x[0], 0.0, 1.0)

        # Joseph form을 사용해 공분산 수치 안정성을 확보
        P = (I3 - K @ H) @ P_pred @ (I3 - K @ H).T + K * R_MEAS * K.T

        soc_hat[k] = x[0]
        voltage_hat[k] = ocv_lut.ocv_value(x[0]) - x[1] - x[2] - params.R0 * Ik

    return {
        "soc_hat": soc_hat,
        "voltage_hat": voltage_hat,
        "voltage_used": voltage_used,
        "slope_tv_weight": slope_tv_weight_hist,
    }


# 8. 성능 평가 지표

def compute_soc_metrics(soc_hat, soc_true):
    """
    SOC 추정 오차 지표 계산

    논문에서는 SOC를 [%] 단위로 표현하기 위해 100을 곱한다.
    """
    soc_hat = np.asarray(soc_hat, dtype=float)
    soc_true = np.asarray(soc_true, dtype=float)

    mask = np.isfinite(soc_hat) & np.isfinite(soc_true)

    err_pct = (soc_hat[mask] - soc_true[mask]) * 100.0

    rmse = np.sqrt(np.mean(err_pct ** 2))
    mae = np.mean(np.abs(err_pct))

    return {
        "RMSE_percent": float(rmse),
        "MAE_percent": float(mae),
    }


# 9. 독립 필터 튜닝 (필터 계수 정할 시 cross validaiotn 사용) 

FILTER_CANDIDATES = {
    "raw": [None],

    "wavelet": [
        {"level": 1},
        {"level": 2},
        {"level": 3},
    ],

    "lowpass": [
        {"cutoff": 0.03, "fs": 1.0},
        {"cutoff": 0.05, "fs": 1.0},
        {"cutoff": 0.10, "fs": 1.0},
    ],

    "savgol": [
        {"window_length": 21},
        {"window_length": 31},
        {"window_length": 51},
    ],

    "fixed_tv": [
        {"weight": 0.10},
        {"weight": 0.15},
        {"weight": 0.20},
    ],

    "slope_tv": [
        {"base_weight": 0.10, "window": 41},
        {"base_weight": 0.15, "window": 41},
        {"base_weight": 0.20, "window": 41},
    ],
}


def run_single_method(record: BatteryRecord,
                      model_params: ModelParams2RC,
                      method,
                      param):
    """
    하나의 배터리 셀에 대해 하나의 방법을 적용한다.
    """
    soc0 = float(record.soc_true[0])

    if method == "slope_tv":
        result = ekf_2rc_soc_estimation(
            current_A=record.current_A,
            voltage_V=record.voltage_V,
            dt=np.ones_like(record.voltage_V) * DT_DEFAULT_SEC,
            ocv_lut=record.ocv_lut,
            Qn_Ah=record.Qn_Ah,
            params=model_params,
            soc0=soc0,
            slope_tv=True,
            slope_tv_base_weight=param["base_weight"],
            slope_tv_window=param["window"],
        )

    else:
        v_input = apply_voltage_filter(record.voltage_V, method, param)

        result = ekf_2rc_soc_estimation(
            current_A=record.current_A,
            voltage_V=v_input,
            dt=np.ones_like(record.voltage_V) * DT_DEFAULT_SEC,
            ocv_lut=record.ocv_lut,
            Qn_Ah=record.Qn_Ah,
            params=model_params,
            soc0=soc0,
            slope_tv=False,
        )

    metrics = compute_soc_metrics(result["soc_hat"], record.soc_true)

    return result, metrics


def tune_filter_parameters(tuning_records, model_param_lookup):
    """
    독립 필터 튜닝 단계

    중요한 점:
    - 이 함수에는 test_records를 절대 넣지 않는다.
    - tuning data에서 평균 RMSE가 가장 낮은 파라미터를 선택한다.
    - 선택된 파라미터는 이후 test data 평가에서 고정하여 사용한다.
    """
    best_params = {}
    tuning_summary = []

    for method, candidates in FILTER_CANDIDATES.items():

        best_rmse = np.inf
        best_param = None

        for param in candidates:
            rmse_list = []

            for record in tuning_records:
                key = (record.dataset, record.cell_id)
                model_params = model_param_lookup[key]

                _, metrics = run_single_method(
                    record=record,
                    model_params=model_params,
                    method=method,
                    param=param,
                )

                rmse_list.append(metrics["RMSE_percent"])

            mean_rmse = float(np.mean(rmse_list))

            tuning_summary.append({
                "method": method,
                "param": param,
                "mean_tuning_RMSE_percent": mean_rmse,
            })

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_param = param

        best_params[method] = best_param

    tuning_summary = pd.DataFrame(tuning_summary)

    return best_params, tuning_summary

# 10. Test data 평가

def evaluate_test_data(test_records, model_param_lookup, best_filter_params):
    """
    Test data evaluation

    tuning 단계에서 선택한 파라미터를 그대로 사용한다.
    따라서 test data의 성능은 파라미터 선정에 영향을 주지 않는다.
    """
    all_results = {}
    metric_rows = []

    for record in test_records:
        key = (record.dataset, record.cell_id)
        model_params = model_param_lookup[key]

        cell_results = {}

        for method, param in best_filter_params.items():
            result, metrics = run_single_method(
                record=record,
                model_params=model_params,
                method=method,
                param=param,
            )

            cell_results[method] = result

            metric_rows.append({
                "dataset": record.dataset,
                "cell_id": record.cell_id,
                "chemistry": record.chemistry,
                "method": method,
                "RMSE_percent": metrics["RMSE_percent"],
                "MAE_percent": metrics["MAE_percent"],
                "MAXE_percent": metrics["MAXE_percent"],
                "selected_param": str(param),
            })

        all_results[key] = cell_results

    metrics_df = pd.DataFrame(metric_rows)

    return all_results, metrics_df

# 11. 결과 시각화

def plot_soc_result(record: BatteryRecord, cell_results, save_path=None):
    """
    SOC 추정 결과와 SOC 오차를 시각화한다.
    """
    t = record.time_s
    soc_true = record.soc_true

    plt.figure(figsize=(8.0, 5.0))

    # SOC estimation
    plt.subplot(2, 1, 1)
    plt.plot(t, soc_true, label="Reference SOC", linewidth=1.4)

    for method, result in cell_results.items():
        plt.plot(t, result["soc_hat"], label=method, linewidth=1.0)

    plt.ylabel("SOC")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8)

    # SOC error
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color="black", linewidth=0.8)

    for method, result in cell_results.items():
        err_pct = (result["soc_hat"] - soc_true) * 100.0
        plt.plot(t, err_pct, label=method, linewidth=1.0)

    plt.xlabel("Time (s)")
    plt.ylabel("SOC Error (%)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# 12. 사용 예시

if __name__ == "__main__":

    """
    실제 사용 시 아래 부분은 사용자의 파일명/폴더 구조에 맞게 수정한다.

    권장 데이터 분리 방식:
    - tuning_records : 필터 파라미터 선정용
    - test_records   : 최종 성능 평가용

    논문에서 강조할 점:
    - tuning_records와 test_records를 분리하여 test-data tuning을 방지하였다.
    - 모든 방법은 동일한 2-RC ECM과 동일한 EKF 설정을 사용하였다.
    - 차이는 EKF 입력 전압 preprocessing 방법에만 있다.
    """

    # -----------------------------------------------------
    # 예시 1: 2-RC 파라미터 테이블
    # 실제로는 익스포넨셜 함수로 계산된 시간 상수로 식별된 R0, R1, C1, R2, C2 값을 CSV/Excel에서 읽어오는 것.
    # -----------------------------------------------------
    model_param_lookup = {
        # ("dataset", cell_id): ModelParams2RC(R0, R1, C1, R2, C2)
        ("4AH_LFP", 1): ModelParams2RC(R0=0.08, R1=0.07, C1=1800, R2=0.02, C2=8000),
        ("4AH_LFP", 2): ModelParams2RC(R0=0.08, R1=0.08, C1=1800, R2=0.02, C2=8000),
    }

    # -----------------------------------------------------
    # 예시 2: BatteryRecord 생성
    # 아래는 형식 예시이다.
    # 실제 분석에서는 read_dst_data(), read_ocv_table()을 이용해 파일에서 읽어오면 된다.
    # -----------------------------------------------------
    # ocv_lut = read_ocv_table("OCV_SOC_Cell1.xlsx")
    # time_s, current_A, voltage_V, soc_true, dt = read_dst_data("DST_Cell1.xlsx")
    # time_s, current_A, voltage_V, soc_true, dt = cut_soc_window(
    #     time_s, current_A, voltage_V, soc_true, dt
    # )
    #
    # record = BatteryRecord(
    #     dataset="4AH_LFP",
    #     cell_id=1,
    #     chemistry="LFP",
    #     Qn_Ah=4.0,
    #     time_s=time_s,
    #     current_A=current_A,
    #     voltage_V=voltage_V,
    #     soc_true=soc_true,
    #     ocv_lut=ocv_lut,
    # )

    # -----------------------------------------------------
    # 예시 3: 전체 실행 흐름
    # -----------------------------------------------------
    # tuning_records = [record_cell_1, record_cell_2, ...]
    # test_records = [record_cell_7, record_cell_8, ...]

    # best_filter_params, tuning_summary = tune_filter_parameters(
    #     tuning_records=tuning_records,
    #     model_param_lookup=model_param_lookup,
    # )

    # all_results, metrics_df = evaluate_test_data(
    #     test_records=test_records,
    #     model_param_lookup=model_param_lookup,
    #     best_filter_params=best_filter_params,
    # )

    # print(tuning_summary)
    # print(metrics_df)

    # 특정 셀 결과 시각화
    # key = ("4AH_LFP", 7)
    # plot_soc_result(
    #     record=test_records[0],
    #     cell_results=all_results[key],
    #     save_path="soc_result_cell7.png",
    # )
