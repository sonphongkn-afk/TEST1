#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_input_excel(path: str, explicit: Optional[str]) -> str:
	if explicit:
		if not os.path.isabs(explicit):
			explicit = os.path.join(path, explicit)
		if not os.path.exists(explicit):
			raise FileNotFoundError(f"Không tìm thấy file: {explicit}")
		return explicit

	candidates: List[str] = []
	for fname in os.listdir(path):
		if not fname.lower().endswith(".xlsx"):
			continue
		if fname.startswith("~$"):
			continue
		lower = fname.lower()
		if any(tag in lower for tag in ["output", "ket_qua", "kq", "result", "summary"]):
			continue
		candidates.append(os.path.join(path, fname))

	if not candidates:
		raise FileNotFoundError("Không tìm thấy file .xlsx đầu vào trong cùng thư mục.")

	# Ưu tiên file có kích thước lớn nhất (thường là file dữ liệu chính)
	candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
	return candidates[0]


def detect_question_columns(df: pd.DataFrame, num_questions: int, first_index: Optional[int]) -> List[str]:
	cols = list(df.columns)
	if first_index is not None:
		end = first_index + num_questions
		if end > len(cols):
			raise ValueError(
				f"first_question_col_index + num_questions vượt quá số cột. {first_index}+{num_questions}>{len(cols)}"
			)
		return cols[first_index:end]

	# Mặc định: lấy 51 cột cuối cùng làm cột câu hỏi
	if len(cols) < num_questions:
		raise ValueError(
			f"File có {len(cols)} cột, ít hơn số câu hỏi yêu cầu ({num_questions}). Hãy kiểm tra file."
		)
	return cols[-num_questions:]


def detect_active_rows(df: pd.DataFrame) -> pd.Series:
	"""Trả về mask (Series bool) hàng hợp lệ (tổng số học sinh).
	Ưu tiên các cột nhận diện như 'Họ và tên', 'Số thứ tự', 'STT'. Nếu không có, dùng hàng có dữ liệu ở bất kỳ cột nào.
	"""
	lower_cols = {str(c).strip().lower(): c for c in df.columns}
	name_like_keys = [
		"họ và tên", "ho va ten", "họ tên", "ho ten", "họ", "ten", "tên", "name",
	]
	id_like_keys = ["số thứ tự", "so thu tu", "stt", "index", "id", "mã", "ma"]

	candidate_cols: List[str] = []
	for key in name_like_keys + id_like_keys:
		if key in lower_cols:
			candidate_cols.append(lower_cols[key])

	mask: pd.Series
	if candidate_cols:
		mask = df[candidate_cols].notna().any(axis=1)
	else:
		mask = df.notna().any(axis=1)

	# Nếu tất cả đều False (toàn bộ trống), coi như tất cả hàng đều hợp lệ để có thể fill dữ liệu
	if not mask.any():
		mask = pd.Series([True] * len(df), index=df.index)
	return mask


def choose_category_counts(num_questions: int, rng: np.random.Generator) -> Tuple[int, int, int]:
	"""Chọn số lượng câu hỏi cho từng dải: 75-80% (1-2), 80-90% (8-10), còn lại 90-<100%.
	Trả về (low_75_80, mid_80_90, high_90_100).
	"""
	low = int(rng.integers(1, 3))  # 1 hoặc 2
	mid = int(rng.integers(8, 11))  # 8..10
	high = num_questions - low - mid
	if high < 0:
		# Trường hợp hiếm khi num_questions quá nhỏ; clamp
		mid = max(0, min(mid, num_questions - low))
		high = num_questions - low - mid
	return low, mid, high


def compute_allowed_range_for_category(N: int, category: str) -> Tuple[int, int]:
	if category == "low":
		return int(np.ceil(0.75 * N)), int(np.floor(0.80 * N))
	if category == "mid":
		return int(np.ceil(0.80 * N)), int(np.floor(0.90 * N))
	# high
	return int(np.ceil(0.90 * N)), max(0, N - 1)


def pick_target_k(
	N: int,
	category: str,
	min_possible: int,
	max_possible: int,
	last_percents: List[float],
	rng: np.random.Generator,
) -> int:
	cat_lo, cat_hi = compute_allowed_range_for_category(N, category)
	lo = max(min_possible, cat_lo)
	hi = min(max_possible, cat_hi)
	if lo > hi:
		# Không thể đạt dải mong muốn, nới lỏng theo khoảng khả dụng
		lo, hi = min_possible, max_possible
		if lo > hi:
			# hoàn toàn bất khả thi (có thể do không còn ô trống), giữ nguyên ở min_possible
			return min_possible

	# Lấy ngẫu nhiên mục tiêu trong [lo, hi]
	candidate_k = int(rng.integers(lo, hi + 1))

	# Tránh trùng lặp phần trăm (làm tròn 2 chữ số) với 1-3 cột trước
	def k_to_percent(k: int) -> float:
		return round((k * 100.0) / max(1, N), 2)

	cand_p = k_to_percent(candidate_k)
	if cand_p not in last_percents:
		return candidate_k

	# Cố gắng điều chỉnh ±1..3
	for delta in [1, -1, 2, -2, 3, -3]:
		k2 = candidate_k + delta
		if k2 < lo or k2 > hi:
			continue
		p2 = k_to_percent(k2)
		if p2 not in last_percents:
			return k2

	# Bất đắc dĩ: quét toàn khoảng để tìm giá trị khác nhau
	for k2 in list(range(lo, hi + 1)):
		p2 = k_to_percent(k2)
		if p2 not in last_percents:
			return k2

	# Không tránh được, trả về candidate ban đầu
	return candidate_k


def fill_column_with_target(
	df: pd.DataFrame,
	col: str,
	active_mask: pd.Series,
	N: int,
	target_k: int,
	rng: np.random.Generator,
) -> None:
	col_series = df.loc[active_mask, col]

	# Chuẩn hóa dữ liệu hiện có về số (1..5), các giá trị ngoài phạm vi sẽ được giữ nguyên để không phá hỏng dữ liệu gốc
	def normalize_value(v):
		try:
			iv = int(float(v))
			if 1 <= iv <= 5:
				return iv
		except Exception:
			pass
		return np.nan

	col_numeric = col_series.apply(normalize_value)
	current_satisfied = int(((col_numeric >= 3) & (col_numeric <= 5)).sum())
	current_unsatisfied = int(((col_numeric >= 1) & (col_numeric <= 2)).sum())
	current_blank_mask = col_numeric.isna()
	blank_indices = col_numeric[current_blank_mask].index.tolist()

	min_possible = current_satisfied
	max_possible = min(N - 1, current_satisfied + len(blank_indices))  # Không cho phép 100%
	if target_k < min_possible:
		target_k = min_possible
	if target_k > max_possible:
		target_k = max_possible

	needed_satisfied = target_k - current_satisfied
	if needed_satisfied < 0:
		needed_satisfied = 0

	# Chọn các ô trống để gán vào nhóm 3-4-5
	if blank_indices:
		if needed_satisfied > 0:
			chosen_for_345 = list(rng.choice(blank_indices, size=needed_satisfied, replace=False))
		else:
			chosen_for_345 = []
		remaining_blanks = [idx for idx in blank_indices if idx not in chosen_for_345]
	else:
		chosen_for_345 = []
		remaining_blanks = []

	# Phân phối 3/4/5 với trọng số biến thiên nhẹ theo từng cột
	w3 = float(rng.uniform(0.15, 0.30))
	w4 = float(rng.uniform(0.30, 0.40))
	w5 = max(0.0, 1.0 - w3 - w4)
	satisfied_choices = rng.choice([3, 4, 5], size=len(chosen_for_345), p=[w3, w4, w5]) if chosen_for_345 else []

	# Phân phối 1/2 cho phần còn lại
	w1 = float(rng.uniform(0.50, 0.70))
	w2 = max(0.0, 1.0 - w1)
	unsatisfied_choices = rng.choice([1, 2], size=len(remaining_blanks), p=[w1, w2]) if remaining_blanks else []

	# Ghi dữ liệu vào df
	for idx, val in zip(chosen_for_345, satisfied_choices):
		df.at[idx, col] = int(val)
	for idx, val in zip(remaining_blanks, unsatisfied_choices):
		df.at[idx, col] = int(val)


def build_summary(df: pd.DataFrame, question_cols: List[str], active_mask: pd.Series) -> pd.DataFrame:
	N = int(active_mask.sum())
	records = []
	for q_idx, col in enumerate(question_cols, start=1):
		series = df.loc[active_mask, col]
		# Chuẩn hóa về số 1..5, giá trị khác coi như NaN (không tính)
		def to_int_1_5(v):
			try:
				iv = int(float(v))
				if 1 <= iv <= 5:
					return iv
			except Exception:
				return np.nan
			return np.nan

		vals = series.apply(to_int_1_5)
		counts = vals.value_counts(dropna=True).reindex([1, 2, 3, 4, 5], fill_value=0)
		percentages = (counts / max(1, N) * 100.0).round(2)
		top345 = float(((counts[3] + counts[4] + counts[5]) / max(1, N) * 100.0).round(2))
		record = {
			"Câu hỏi": f"Câu {q_idx}",
			"Cột": col,
			"SL mức 1": int(counts[1]),
			"SL mức 2": int(counts[2]),
			"SL mức 3": int(counts[3]),
			"SL mức 4": int(counts[4]),
			"SL mức 5": int(counts[5]),
			"% mức 1": float(percentages[1]),
			"% mức 2": float(percentages[2]),
			"% mức 3": float(percentages[3]),
			"% mức 4": float(percentages[4]),
			"% mức 5": float(percentages[5]),
			"MỨC ĐỘ HOÀN TOÀN HÀI LÒNG (3-5) %": top345,
		}
		records.append(record)
	return pd.DataFrame.from_records(records)


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Đọc file Excel .xlsx, fill các ô trống trong 51 cột câu hỏi bằng số 1..5 "
			"theo kiểm soát ngẫu nhiên, và xuất file kết quả với bảng tổng hợp (không có 100%, "
			"tránh trùng lặp 1-3 cột liền kề, và phân phối 75-80/80-90/90-<100 như yêu cầu)."
		)
	)
	parser.add_argument("--input", type=str, default=None, help="Tên file .xlsx đầu vào (mặc định: tự động tìm trong thư mục hiện tại)")
	parser.add_argument("--output", type=str, default=None, help="Tên file .xlsx kết quả (mặc định: <input>_ket_qua.xlsx)")
	parser.add_argument("--questions", type=int, default=51, help="Số lượng câu hỏi (mặc định: 51)")
	parser.add_argument("--first_question_col_index", type=int, default=None, help="Chỉ số cột bắt đầu cho khối câu hỏi (0-based). Mặc định: dùng 51 cột cuối cùng")
	parser.add_argument("--seed", type=int, default=None, help="Seed ngẫu nhiên để tái lập kết quả")
	args = parser.parse_args()

	cwd = os.getcwd()

	try:
		input_path = detect_input_excel(cwd, args.input)
	except Exception as e:
		print(str(e))
		return 1

	if args.output:
		output_path = args.output if os.path.isabs(args.output) else os.path.join(cwd, args.output)
	else:
		base, ext = os.path.splitext(os.path.basename(input_path))
		output_path = os.path.join(cwd, f"{base}_ket_qua.xlsx")

	rng = np.random.default_rng(args.seed)

	# Đọc dữ liệu
	try:
		df = pd.read_excel(input_path, engine="openpyxl")
	except Exception as e:
		print(f"Lỗi khi đọc file Excel: {e}")
		return 1

	# Xác định cột câu hỏi và hàng hợp lệ
	try:
		question_cols = detect_question_columns(df, args.questions, args.first_question_col_index)
	except Exception as e:
		print(str(e))
		return 1

	active_mask = detect_active_rows(df)
	N = int(active_mask.sum()) if int(active_mask.sum()) > 0 else len(df)

	# Lập danh sách hạng mục cho 51 câu hỏi
	low, mid, high = choose_category_counts(len(question_cols), rng)
	categories = ["low"] * low + ["mid"] * mid + ["high"] * high
	rng.shuffle(categories)

	# Duyệt từng cột để ấn định target_k và fill dữ liệu
	last_percents: List[float] = []  # lưu các % (2 chữ số) của 1..3 cột trước

	for idx, col in enumerate(question_cols):
		category = categories[idx] if idx < len(categories) else "high"
		col_series = df.loc[active_mask, col]

		# Chuẩn hóa hiện trạng
		def normalize_value(v):
			try:
				iv = int(float(v))
				if 1 <= iv <= 5:
					return iv
			except Exception:
				return np.nan
			return np.nan

		col_numeric = col_series.apply(normalize_value)
		current_satisfied = int(((col_numeric >= 3) & (col_numeric <= 5)).sum())
		blank_cnt = int(col_numeric.isna().sum())
		min_possible = current_satisfied
		max_possible = min(N - 1, current_satisfied + blank_cnt)

		target_k = pick_target_k(N, category, min_possible, max_possible, last_percents, rng)
		fill_column_with_target(df, col, active_mask, N, target_k, rng)

		# Cập nhật last_percents
		percent = round((max(target_k, current_satisfied) * 100.0) / max(1, N), 2)
		last_percents.append(percent)
		if len(last_percents) > 3:
			last_percents = last_percents[-3:]

	# Xây bảng tổng hợp
	summary_df = build_summary(df, question_cols, active_mask)

	# Ghi ra Excel
	try:
		with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
			# Sheet 1: Dữ liệu đã fill
			df.to_excel(writer, index=False, sheet_name="DATA_FILLED")

			# Sheet 2: Tổng hợp
			summary_df.to_excel(writer, index=False, sheet_name="SUMMARY")

			# Định dạng cột phần trăm
			workbook = writer.book
			pct_fmt = workbook.add_format({"num_format": "0.00"})  # hiển thị dạng 0.00 (đã là %) trên thang 0-100

			ws_sum = writer.sheets["SUMMARY"]
			# Tìm các cột phần trăm theo tiêu đề
			for col_idx, col_name in enumerate(summary_df.columns):
				if str(col_name).strip().startswith("% ") or "(3-5) %" in str(col_name):
					ws_sum.set_column(col_idx, col_idx, 14, pct_fmt)
				else:
					ws_sum.set_column(col_idx, col_idx, 18)

			ws_data = writer.sheets["DATA_FILLED"]
			ws_data.freeze_panes(1, 1)
			ws_sum.freeze_panes(1, 1)

			# Ghi sheet tham số
			params_df = pd.DataFrame(
				{
					"Tham số": [
						"Tổng số học sinh (N)",
						"Số câu 75-80%",
						"Số câu 80-90%",
						"Số câu 90-<100%",
						"Seed",
					],
					"Giá trị": [N, low, mid, high, args.seed if args.seed is not None else "(ngẫu nhiên)"],
				}
			)
			params_df.to_excel(writer, index=False, sheet_name="PARAMS")

		print(f"Đã tạo file kết quả: {output_path}")
	except Exception as e:
		print(f"Lỗi khi ghi file Excel: {e}")
		return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())