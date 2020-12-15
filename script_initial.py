
import numpy as np
import matplotlib.pyplot as plt


class SudokuBoard(object):

    def __init__(self, initial_state):

        self._fixed = np.zeros((9, 9, 9), np.bool)
        self._candidates = np.zeros((9, 9, 9), np.bool)
        self._locked = np.zeros((9, 9), np.bool)
        self._init_board(initial_state=initial_state)
        self._plot_num = plt.figure(figsize=(8, 8)).number


    def _init_board(self, initial_state):
        idx_row, idx_col = np.where(initial_state > 0)
        values = initial_state[idx_row, idx_col] - 1
        self._fixed[idx_row, idx_col, values] = True

    def solve(self):
        self.show()
        self._candidates = self._init_candidates()
        self.show()

        while not self._is_solved():
            success = self.iterate()
            if not success:
                print('could not find any more. :-(')
                break
            self.show()

    def _is_solved(self):
        return self._fixed.any(axis=2).all()

    def iterate(self):
        already_found = np.where(self._candidates.sum(axis=2) == 1)
        if len(already_found[0]) > 0:
            row = already_found[0][0]
            col = already_found[1][0]
            value = np.where(self._candidates[row, col])[0][0] + 1
            self.set_new_value(row, col, value)
            return True
        if self._find_locked_tuples():
            return True

        return False

    def _find_locked_tuples(self):
        found_locked_tuple = False
        for row in range(9):
            for col in range(9):
                if self._fixed[row, col].any():
                    continue
                if self._locked[row, col]:
                    continue
                local_candidates = self._candidates[row:row+1, col:col+1]
                lock = (local_candidates == self._candidates).all(axis=2)
                local_count = local_candidates.sum()
                locked_in_row = lock[row, :]
                locked_in_col = lock[:, col]
                idx_block_x = col // 3
                idx_block_y = row // 3
                slice_x = slice(idx_block_x * 3, (idx_block_x + 1) * 3, None)
                slice_y = slice(idx_block_y * 3, (idx_block_y + 1) * 3, None)
                locked_in_block = lock[slice_y, slice_x]
                if np.sum(locked_in_row) == local_count:
                    locked_cols = np.where(locked_in_row)[0]
                    unbound_cols = np.where(np.logical_not(locked_in_row))[0]
                    locked_values = np.where(local_candidates.squeeze())[0]
                    locked_rows = np.ones(unbound_cols.shape[0], dtype=int)*row
                    self._locked[row, locked_cols] = True
                    found_locked_tuple = True
                    self._remove_candidates(
                        rows=locked_rows,
                        cols=unbound_cols,
                        values=locked_values
                    )
                    print(':-)')
                if np.sum(locked_in_col) == local_count:
                    print(':-)')
                    # raise NotImplementedError
                if np.sum(locked_in_block) == local_count:
                    print(':-)')
                    # raise NotImplementedError
                if found_locked_tuple:
                    return found_locked_tuple
        return found_locked_tuple

    def _remove_candidates(self, rows, cols, values):
        for value in values:
            self._candidates[rows, cols, value] = False


    def set_new_value(self, row, col, value):
        idx_val = value-1
        self._fixed[row, col, idx_val] = True
        self._candidates[row, :, idx_val] = False
        self._candidates[:, col, idx_val] = False
        idx_block_x = col // 3
        idx_block_y = row // 3
        slice_x = slice(idx_block_x * 3, (idx_block_x + 1) * 3, None)
        slice_y = slice(idx_block_y * 3, (idx_block_y + 1) * 3, None)
        self._candidates[slice_y, slice_x, idx_val] = False

    def show(self):
        # draw lines
        plt.figure(num=self._plot_num)
        plt.clf()
        self._draw_empty_board()
        self._draw_fixed()
        self._draw_candidates()

    def _draw_candidates(self):
        for idx_row in range(9):
            for idx_col in range(9):
                if self._fixed[idx_row, idx_col].any():
                    continue
                values = np.where(self._candidates[idx_row, idx_col])[0] + 1
                pos_x = idx_col + 0.5
                pos_y = idx_row + 0.5
                text = ''
                for idx, v in enumerate(values, start=1):
                    text += str(v)
                    if (idx%3 == 0) and (idx != len(values)):
                        text += '\n'
                if self._locked[idx_row, idx_col]:
                    color = 'blue'
                else:
                    color = 'gray'
                plt.text(pos_x, pos_y, text, fontsize=16,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color=color)

    def _init_candidates(self):
        candidates = np.ones((9, 9, 9), np.bool)
        for idx_row in range(9):
            for idx_col in range(9):
                if self._fixed[idx_row, idx_col].any():
                    candidates[idx_row, idx_col] = False
                    continue
                # find all digits already used
                in_row = np.where(self._fixed[idx_row].any(axis=0))[0] + 1
                in_col = np.where(self._fixed[:, idx_col].any(axis=0))[0] + 1
                idx_block_x = idx_col // 3
                idx_block_y = idx_row // 3
                slice_x = slice(idx_block_x*3, (idx_block_x+1)*3, None)
                slice_y = slice(idx_block_y*3, (idx_block_y+1)*3, None)
                block = self._fixed[slice_y, slice_x]
                in_block = np.where(block.any(axis=(0, 1)))[0] + 1
                already_set = np.unique(np.r_[in_row, in_col, in_block])
                candidates[idx_row, idx_col, already_set-1] = False
        return candidates

    def _draw_fixed(self):
        for idx_row, idx_col, values in zip(*np.where(self._fixed)):
            pos_x = idx_col+0.5
            pos_y = idx_row+0.5
            plt.text(pos_x, pos_y, values+1, fontsize=48,
                     verticalalignment='center',
                     horizontalalignment='center'
            )


    @staticmethod
    def _draw_empty_board():
        for offset in range(10):
            if offset % 3 == 0:
                line_width = 3
            else:
                line_width = 1
            plt.plot([0, 9], [offset, offset], 'k', linewidth=line_width)
            plt.plot([offset, offset], [0, 9], 'k', linewidth=line_width)
        plt.axis('off')
        plt.ylim([9.2, -0.2])
        plt.xlim([-0.2, 9.2])
        plt.tight_layout()



if __name__ == '__main__':
    initial_state = np.asarray([
        [6, 9, 0, 0, 8, 0, 0, 0, 0],
        [0, 8, 2, 0, 6, 5, 0, 7, 0],
        [0, 0, 0, 0, 0, 1, 0, 8, 3],
        [0, 3, 6, 0, 0, 0, 5, 0, 0],
        [9, 0, 0, 0, 5, 0, 0 ,0, 2],
        [0, 0, 5, 0, 0, 0, 1, 6, 0],
        [2, 5, 0, 6, 0, 0, 0, 0, 0],
        [0, 6, 0, 3, 1, 0, 4, 9, 0],
        [0, 0, 0, 0, 9, 0, 0, 2, 6]
    ])

    board = SudokuBoard(
        initial_state=initial_state
    )
    board.solve()