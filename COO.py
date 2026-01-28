from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List

# используется для корректной проверки типов
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSC import CSCMatrix
    from CSR import CSRMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        # проверка согласованности входных данных
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Количество элементов в data, row, col должно совпадать")

        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Конвертирует разреженную COO‑матрицу в плотный формат."""
        n_rows, n_cols = self.shape
        result = [[0] * n_cols for _ in range(n_rows)]

        for val, r, c in zip(self.data, self.row, self.col):
            result[r][c] = val

        return result

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения двух COO‑матриц."""
        # используем словарь для накопления сумм
        accumulator: Dict[Tuple[int, int], float] = {}

        # добавляем элементы первой матрицы
        for v, r, c in zip(self.data, self.row, self.col):
            accumulator[(r, c)] = accumulator.get((r, c), 0.0) + v

        # добавляем элементы второй матрицы
        for v, r, c in zip(other.data, other.row, other.col):
            accumulator[(r, c)] = accumulator.get((r, c), 0.0) + v

        # собираем только ненулевые элементы
        new_data, new_rows, new_cols = [], [], []
        for (r, c), total in accumulator.items():
            if abs(total) > 1e-14:
                new_data.append(total)
                new_rows.append(r)
                new_cols.append(c)

        return COOMatrix(new_data, new_rows, new_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на число."""
        scaled_data = [v * scalar for v in self.data]
        return COOMatrix(scaled_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Возвращает транспонированную матрицу."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц для умножения")

        m, n = self.shape[0], other.shape[1]
        temp_result = {}

        # конвертируем правую матрицу в CSR для эффективного доступа
        other_csr = other._to_csr()

        for idx in range(len(self.data)):
            r = self.row[idx]
            c = self.col[idx]
            v_a = self.data[idx]

            start = other_csr.indptr[c]
            stop = other_csr.indptr[c + 1]

            for pos in range(start, stop):
                col_b = other_csr.indices[pos]
                v_b = other_csr.data[pos]
                key = (r, col_b)
                temp_result[key] = temp_result.get(key, 0.0) + v_a * v_b

        # преобразуем результат обратно в COO
        res_data, res_rows, res_cols = [], [], []
        for (r, c), val in temp_result.items():
            if abs(val) > 1e-14:
                res_data.append(val)
                res_rows.append(r)
                res_cols.append(c)

        return COOMatrix(res_data, res_rows, res_cols, (m, n))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'COOMatrix':
        """Создаёт COO‑матрицу из плотного представления."""
        data, rows, cols = [], [], []
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0

        for r in range(n_rows):
            for c in range(n_cols):
                elem = dense[r][c]
                if elem != 0:
                    data.append(elem)
                    rows.append(r)
                    cols.append(c)

        return cls(data, rows, cols, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует текущую матрицу в формат CSC."""
        from CSC import CSCMatrix
        n_rows, n_cols = self.shape

        # сортируем элементы по столбцам, затем по строкам
        entries = list(zip(self.col, self.row, self.data))
        entries.sort()

        csc_data = []
        csc_indices = []
        csc_indptr = [0] * (n_cols + 1)

        for col, row, val in entries:
            csc_data.append(val)
            csc_indices.append(row)
            csc_indptr[col + 1] += 1

        # аккумулируем указатели столбцов
        for j in range(n_cols):
            csc_indptr[j + 1] += csc_indptr[j]

        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует текущую матрицу в формат CSR."""
        from CSR import CSRMatrix
        n_rows, n_cols = self.shape

        # сортируем элементы по строкам, затем по столбцам
        entries = list(zip(self.row, self.col, self.data))
        entries.sort()

        csr_data = []
        csr_indices = []
        csr_indptr = [0] * (n_rows + 1)

        for row, col, val in entries:
            csr_data.append(val)
            csr_indices.append(col)
            csr_indptr[row + 1] += 1

        # аккумулируем указатели строк
        for i in range(n_rows):
            csr_indptr[i + 1] += csr_indptr[i]

        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)