from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

# используется для корректной проверки типов
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)

        # проверки корректности структуры CSC
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[1] + 1}")
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
        """Преобразует CSC‑матрицу в плотный формат."""
        n_rows, n_cols = self.shape
        dense = [[0] * n_cols for _ in range(n_rows)]

        for col in range(n_cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                val = self.data[idx]
                dense[row][col] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение двух CSC‑матриц через промежуточное COO‑представление."""
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на скаляр."""
        if abs(scalar) < 1e-14:
            # результат – нулевая матрица той же размерности
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        scaled_data = [v * scalar for v in self.data]
        return CSCMatrix(scaled_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы. Результат возвращается в формате CSR."""
        from CSR import CSRMatrix
        old_rows, old_cols = self.shape
        new_rows, new_cols = old_cols, old_rows

        # подсчёт количества элементов в каждой строке результата
        row_sizes = [0] * new_rows
        for col in range(old_cols):
            row_sizes[col] = self.indptr[col + 1] - self.indptr[col]

        new_indptr = [0] * (new_rows + 1)
        for i in range(new_rows):
            new_indptr[i + 1] = new_indptr[i] + row_sizes[i]

        result_data = [0] * len(self.data)
        result_indices = [0] * len(self.indices)
        fill_positions = new_indptr.copy()

        for col in range(old_cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                val = self.data[idx]

                pos = fill_positions[col]
                result_data[pos] = val
                result_indices[pos] = row
                fill_positions[col] += 1

        return CSRMatrix(result_data, result_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC‑матрицы на другую матрицу."""
        m, n = self.shape[0], other.shape[1]

        result_data = []
        result_indices = []
        result_indptr = [0] * (n + 1)

        # предподготовка данных второй матрицы по строкам
        rows_of_B = [[] for _ in range(other.shape[0])]
        for col in range(n):
            start = other.indptr[col]
            end = other.indptr[col + 1]
            for idx in range(start, end):
                row = other.indices[idx]
                val = other.data[idx]
                rows_of_B[row].append((col, val))

        # временный массив для накопления строки результата
        temp_row = [0.0] * m

        for j in range(n):
            # обнуляем временный массив
            for i in range(m):
                temp_row[i] = 0.0

            # вычисляем j‑й столбец результата
            for i in range(other.shape[0]):
                for col_b, val_b in rows_of_B[i]:
                    if col_b == j:
                        col_start = self.indptr[i]
                        col_end = self.indptr[i + 1]
                        for a_idx in range(col_start, col_end):
                            row_a = self.indices[a_idx]
                            val_a = self.data[a_idx]
                            temp_row[row_a] += val_a * val_b

            # сохраняем ненулевые элементы столбца
            for i in range(m):
                if abs(temp_row[i]) > 1e-14:
                    result_data.append(temp_row[i])
                    result_indices.append(i)

            result_indptr[j + 1] = len(result_data)

        return CSCMatrix(result_data, result_indices, result_indptr, (m, n))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSCMatrix':
        """Создаёт CSC‑матрицу из плотного представления."""
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0

        data = []
        indices = []
        col_counts = [0] * n_cols

        for c in range(n_cols):
            for r in range(n_rows):
                elem = dense[r][c]
                if elem != 0:
                    data.append(elem)
                    indices.append(r)
                    col_counts[c] += 1

        indptr = [0] * (n_cols + 1)
        for j in range(n_cols):
            indptr[j + 1] = indptr[j] + col_counts[j]

        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует CSC‑матрицу в формат CSR."""
        from CSR import CSRMatrix
        m, n = self.shape

        # подсчёт количества элементов в каждой строке
        row_counts = [0] * m
        for r in self.indices:
            row_counts[r] += 1

        indptr = [0] * (m + 1)
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]

        data = [0] * len(self.data)
        indices = [0] * len(self.indices)
        insert_pos = indptr.copy()

        for j in range(n):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                val = self.data[idx]

                pos = insert_pos[i]
                data[pos] = val
                indices[pos] = j
                insert_pos[i] += 1

        return CSRMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        """Конвертирует CSC‑матрицу в формат COO."""
        from COO import COOMatrix
        n_rows, n_cols = self.shape

        coo_data = []
        coo_rows = []
        coo_cols = []

        for col in range(n_cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                val = self.data[idx]
                coo_data.append(val)
                coo_rows.append(row)
                coo_cols.append(col)

        return COOMatrix(coo_data, coo_rows, coo_cols, self.shape)