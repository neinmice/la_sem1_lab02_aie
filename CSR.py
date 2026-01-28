from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

# используется для корректной проверки типов
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        # проверки корректности структуры CSR
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[0] + 1}")
        if indptr[0] != 0:
            raise ValueError("Первый элемент indptr должен быть равен 0")
        if indptr[-1] != len(data):
            raise ValueError(f"Последний элемент indptr должен быть равен {len(data)}")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")

        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR‑матрицу в плотный формат."""
        n_rows, n_cols = self.shape
        dense = [[0] * n_cols for _ in range(n_rows)]

        for i in range(n_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                val = self.data[idx]
                dense[i][j] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение двух CSR‑матриц через промежуточное COO‑представление."""
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на скаляр."""
        if abs(scalar) < 1e-14:
            # результат – нулевая матрица той же размерности
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        scaled_data = [v * scalar for v in self.data]
        return CSRMatrix(scaled_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы. Результат возвращается в формате CSC."""
        from CSC import CSCMatrix
        old_rows, old_cols = self.shape
        new_rows, new_cols = old_cols, old_rows

        # подсчёт количества элементов в каждом столбце результата
        col_sizes = [0] * new_cols
        for i in range(old_rows):
            col_sizes[i] = self.indptr[i + 1] - self.indptr[i]

        new_indptr = [0] * (new_cols + 1)
        for j in range(new_cols):
            new_indptr[j + 1] = new_indptr[j] + col_sizes[j]

        result_data = [0] * len(self.data)
        result_indices = [0] * len(self.indices)
        fill_positions = new_indptr.copy()

        for i in range(old_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                val = self.data[idx]

                pos = fill_positions[i]
                result_data[pos] = val
                result_indices[pos] = j
                fill_positions[i] += 1

        return CSCMatrix(result_data, result_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR‑матрицы на другую матрицу."""
        m, n = self.shape[0], other.shape[1]

        result_data = []
        result_indices = []
        result_indptr = [0] * (m + 1)

        for i in range(m):
            row_accumulator = {}

            a_start = self.indptr[i]
            a_end = self.indptr[i + 1]

            for a_idx in range(a_start, a_end):
                k = self.indices[a_idx]
                a_val = self.data[a_idx]

                b_start = other.indptr[k]
                b_end = other.indptr[k + 1]

                for b_idx in range(b_start, b_end):
                    j = other.indices[b_idx]
                    b_val = other.data[b_idx]

                    row_accumulator[j] = row_accumulator.get(j, 0.0) + a_val * b_val

            # сохраняем ненулевые элементы строки результата
            sorted_columns = sorted(row_accumulator.keys())
            for j in sorted_columns:
                val = row_accumulator[j]
                if abs(val) > 1e-14:
                    result_data.append(val)
                    result_indices.append(j)

            result_indptr[i + 1] = len(result_data)

        return CSRMatrix(result_data, result_indices, result_indptr, (m, n))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSRMatrix':
        """Создаёт CSR‑матрицу из плотного представления."""
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0

        data = []
        indices = []
        row_counts = [0] * n_rows

        for r in range(n_rows):
            for c in range(n_cols):
                elem = dense[r][c]
                if elem != 0:
                    data.append(elem)
                    indices.append(c)
                    row_counts[r] += 1

        indptr = [0] * (n_rows + 1)
        for i in range(n_rows):
            indptr[i + 1] = indptr[i] + row_counts[i]

        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует CSR‑матрицу в формат CSC."""
        from CSC import CSCMatrix
        m, n = self.shape

        # подсчёт количества элементов в каждом столбце
        col_counts = [0] * n
        for c in self.indices:
            col_counts[c] += 1

        indptr = [0] * (n + 1)
        for j in range(n):
            indptr[j + 1] = indptr[j] + col_counts[j]

        data = [0] * len(self.data)
        indices = [0] * len(self.indices)
        insert_pos = indptr.copy()

        for i in range(m):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                val = self.data[idx]

                pos = insert_pos[j]
                data[pos] = val
                indices[pos] = i
                insert_pos[j] += 1

        return CSCMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        """Конвертирует CSR‑матрицу в формат COO."""
        from COO import COOMatrix
        n_rows, n_cols = self.shape

        coo_data = []
        coo_rows = []
        coo_cols = []

        for i in range(n_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                val = self.data[idx]
                coo_data.append(val)
                coo_rows.append(i)
                coo_cols.append(j)

        return COOMatrix(coo_data, coo_rows, coo_cols, self.shape)