import json
import numpy as np
import matplotlib.pyplot as plt


class SudokuBoard(object):

    def __init__(self, initial_state):

        self._fixed = np.zeros((9, 9, 9), np.bool)
        self._candidates = np.zeros((9, 9, 9), np.bool)
        self._locked = np.zeros((9, 9), np.bool)
        self._init_board(initial_state=initial_state)
        self._plot_num = plt.figure(figsize=(8, 8)).number
        self._iteration = 0

    def _init_board(self, initial_state):
        idx_row, idx_col = np.where(initial_state > 0)
        values = initial_state[idx_row, idx_col] - 1
        self._fixed[idx_row, idx_col, values] = True

    def solve(self):
        self.show()
        self._candidates = self._init_candidates()
        self.show()

        while not self._is_solved():
            print(f'iteration {self._iteration}')
            self._sanity_check()
            success = self.iterate()
            if not success:
                print('could not find any more. :-(')
                plt.show()
                return
            self.show()
            self._iteration += 1
        print(f'solved in {self._iteration} iterations :-)')
        plt.show()

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
        # check single digits in block
        if self._check_singles_in_block():
            return True

        # check for single digits in column
        if self._check_singles_in_col():
            return True

        # check for single digits in row
        if self._check_singles_in_row():
            return True

        if self._find_locked_tuples():
            return True

        if self._clean_row_col_locking():
            return True

        return False

    def _clean_row_col_locking(self):
        found = False
        for block_y in range(3):
            slice_y = slice(block_y*3, (block_y+1)*3, None)
            for block_x in range(3):
                slice_x = slice(block_x * 3, (block_x + 1) * 3, None)
                block = self._candidates[slice_y, slice_x]
                locked_in_row = np.where(block[:, :].any(axis=1).sum(axis=0)
                                         == 1)[0]
                for locked_value in locked_in_row:
                    blocked = block[:, :, locked_value].any(axis=1)
                    locked_row = np.where(blocked)[0][0]
                    idx_row = locked_row + block_y*3
                    mask = np.ones(9, np.bool)
                    mask[slice_x] = False
                    if np.any(self._candidates[idx_row, mask, locked_value]):
                        self._candidates[idx_row, mask, locked_value] = False
                        found = True
                blocked = block[:, :].any(axis=0).sum(axis=1) == 1
                locked_in_col = np.where(blocked)[0]
                for locked_value in locked_in_col:
                    blocked = block[:, :, locked_value].any(axis=0)
                    locked_col = np.where(blocked)[0][0]
                    idx_col = locked_col + block_x * 3
                    mask = np.ones(9, np.bool)
                    mask[slice_y] = False
                    if np.any(self._candidates[mask, idx_col, locked_value]):
                        self._candidates[mask, idx_col, locked_value] = False
                        found = True
        return found

    def _check_singles_in_block(self):
        for block_y in range(3):
            slice_y = slice(block_y*3, (block_y+1)*3, None)
            for block_x in range(3):
                slice_x = slice(block_x * 3, (block_x + 1) * 3, None)
                candidate_block = self._candidates[slice_y, slice_x]
                values = np.where(candidate_block.sum(axis=(0, 1)) == 1)[0]
                if len(values):
                    value = values[0]
                    row, col = np.where(candidate_block[:, :, value])
                    idx_row = row[0] + block_y*3
                    idx_col = col[0] + block_x*3
                    self.set_new_value(row=idx_row, col=idx_col, value=value+1)
                    return True
        return False

    def _check_singles_in_col(self):
        singles_in_col = np.where(self._candidates.sum(axis=0) == 1)
        found = len(singles_in_col[0])
        for col, value in zip(*singles_in_col):
            row = np.where(self._candidates[:, col, value])[0][0]
            self.set_new_value(row=row, col=col, value=value+1)
            break
        return found

    def _check_singles_in_row(self):
        singles_in_row = np.where(self._candidates.sum(axis=1) == 1)
        found = len(singles_in_row[0])
        for row, value in zip(*singles_in_row):
            col = np.where(self._candidates[row, :, value])[0][0]
            self.set_new_value(row=row, col=col, value=value+1)
            break
        return found

    def _sanity_check(self):
        if (self._fixed.sum(axis=0) > 1).any():
            raise Exception('riddle in bad state: same digits in a row')
        if (self._fixed.sum(axis=1) > 1).any():
            raise Exception('riddle in bad state: same digits in a col')
        if (self._fixed.sum(axis=2) > 1).any():
            raise Exception('riddle in bad state: more than of fixed number')

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
                locked_values = np.where(local_candidates.squeeze())[0]
                if np.sum(locked_in_row) == local_count:
                    locked_cols = np.where(locked_in_row)[0]
                    unbound_cols = np.where(np.logical_not(locked_in_row))[0]
                    # locked_values = np.where(local_candidates.squeeze())[0]
                    locked_rows = np.ones(unbound_cols.shape[0], dtype=int)*row
                    self._locked[row, locked_cols] = True
                    found_locked_tuple = True
                    self._remove_candidates(
                        rows=locked_rows,
                        cols=unbound_cols,
                        values=locked_values
                    )
                if np.sum(locked_in_col) == local_count:
                    locked_rows = np.where(locked_in_col)[0]
                    unbound_rows = np.where(np.logical_not(locked_in_col))[0]
                    # locked_values = np.where(local_candidates.squeeze())[0]
                    locked_cols = np.ones(unbound_rows.shape[0],
                                          dtype=int) * col
                    self._locked[locked_rows, col] = True
                    found_locked_tuple = True
                    self._remove_candidates(
                        rows=unbound_rows,
                        cols=locked_cols,
                        values=locked_values
                    )
                    # raise NotImplementedError
                if np.sum(locked_in_block) == local_count:
                    locked_rows, locked_cols = np.where(locked_in_block)
                    locked_rows += slice_y.start
                    locked_cols += slice_x.start
                    self._locked[locked_rows, locked_cols] = True
                    not_locked_in_block = np.logical_not(locked_in_block)
                    unbound_rows, unbound_cols = np.where(not_locked_in_block)
                    unbound_rows += slice_y.start
                    unbound_cols += slice_x.start
                    # raise Exception('check locked_values')
                    self._remove_candidates(
                        rows=unbound_rows,
                        cols=unbound_cols,
                        values=locked_values,
                    )
                if found_locked_tuple:
                    return found_locked_tuple
        return found_locked_tuple

    def _remove_candidates(self, rows, cols, values):
        for value in values:
            self._candidates[rows, cols, value] = False

    def set_new_value(self, row, col, value):
        idx_val = value-1
        self._fixed[row, col, idx_val] = True
        self._candidates[row, col, :] = False
        self._candidates[row, :, idx_val] = False
        self._candidates[:, col, idx_val] = False
        slice_y, slice_x = self._get_slices(row=row, col=col)
        self._candidates[slice_y, slice_x, idx_val] = False

    @staticmethod
    def _get_slices(row, col):
        idx_block_x = col // 3
        idx_block_y = row // 3
        slice_x = slice(idx_block_x * 3, (idx_block_x + 1) * 3, None)
        slice_y = slice(idx_block_y * 3, (idx_block_y + 1) * 3, None)
        return slice_y, slice_x

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
                    if (idx % 3 == 0) and (idx != len(values)):
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
            plt.text(
                x=pos_x,
                y=pos_y,
                s=values+1,
                fontsize=48,
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
    path_sample = './samples/sample_expert.json'
    path_sample = './samples/sample_easy_2.json'
    with open(path_sample, 'r') as file:
        initial_state = np.asarray(json.load(file))

    board = SudokuBoard(
        initial_state=initial_state
    )
    board.solve()
