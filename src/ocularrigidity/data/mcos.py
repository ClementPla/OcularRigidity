"""Read MATLAB class objects (MCOS) saved in a v5 MAT-file."""

import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader

__all__ = ["MCOSFile", "ObjectArray"]

# First word of the uint32 array that stands in for an object reference.
_REFERENCE_MAGIC = 0xDD000000


class ObjectArray:
    """An array of objects, each decoded to a dict of its properties on access."""

    def __init__(self, file: "MCOSFile", object_ids):
        self._file = file
        self._ids = [int(i) for i in object_ids]

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> dict:
        return self._file.object(self._ids[index])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        return f"<ObjectArray of {len(self)}>"


class MCOSFile:
    def __init__(self, path: str | Path):
        mat = loadmat(path, squeeze_me=False, struct_as_record=False)
        self._top = {k: v for k, v in mat.items() if not k.startswith("__")}

        raw = np.asarray(mat["__function_workspace__"]).tobytes()
        buf = io.BytesIO(b" " * 116 + b"\x00" * 8 + b"\x00\x01" + b"IM" + raw[8:])
        reader = MatFile5Reader(buf, struct_as_record=False, squeeze_me=False)
        reader.initialize_read()
        buf.seek(128)
        header, _ = reader.read_var_header()
        wrapper = reader.read_var_array(header, process=False)[0, 0].MCOS
        self._cells = np.asarray(wrapper[0]["arr"]).ravel()

        meta = np.asarray(self._cells[0]).ravel().tobytes()
        _, n_names = np.frombuffer(meta[:8], "<u4")
        offsets = np.frombuffer(meta[8:40], "<u4")

        def words(a, b):
            return np.frombuffer(meta[a:b], "<u4")

        self.names = [n.decode() for n in meta[40 : offsets[0]].split(b"\x00")[:n_names]]
        self._classes = words(offsets[0], offsets[1]).reshape(-1, 4)
        # Per object: class id, 0, 0, id in the first property table, id in the
        # second, dependency id.
        self._objects = words(offsets[2], offsets[3]).reshape(-1, 6)
        self._tables = (
            self._property_blocks(words(offsets[1], offsets[2])),
            self._property_blocks(words(offsets[3], offsets[4])),
        )

    @staticmethod
    def _property_blocks(words: np.ndarray) -> list:
        """Split a property table into one ``(n, 3)`` block per object."""
        blocks, i = [np.empty((0, 3), dtype=np.uint32)], 2
        while i < len(words):
            n = int(words[i])
            i += 1
            blocks.append(words[i : i + 3 * n].reshape(n, 3))
            i += 3 * n
            i += i % 2
        return blocks

    def object(self, object_id: int) -> dict:
        """Properties of one object, plus its class name under ``__class__``."""
        class_id, _, _, first, second, _ = self._objects[object_id]
        block = self._tables[0][first] if first else self._tables[1][second]
        out = {"__class__": self.names[self._classes[class_id][1] - 1]}
        for name, kind, value in block:
            key = self.names[name - 1]
            if kind == 1:  # index into the value cells
                out[key] = self._resolve(self._cells[2 + value])
            elif kind == 0:  # the value is itself a name
                out[key] = self.names[value - 1]
            else:  # logical
                out[key] = bool(value)
        return out

    def _resolve(self, value):
        """Turn an object reference into an :class:`ObjectArray`"""
        ids = _reference_ids(value)
        return value if ids is None else ObjectArray(self, ids)

    def root(self) -> dict:
        """The (single) object saved as the file's top-level variable."""
        if len(self._top) != 1:
            raise ValueError(f"Expected one top-level variable, got {list(self._top)}")
        (handle,) = self._top.values()
        ids = _reference_ids(handle[0]["arr"])
        if ids is None or len(ids) != 1:
            raise ValueError("Top-level variable is not a single MCOS object")
        return self.object(int(ids[0]))


def _reference_ids(value):
    """Object ids if ``value`` is an object reference, else None."""
    a = np.asarray(value)
    if a.dtype != np.uint32 or a.size < 6:
        return None
    w = a.ravel()
    if w[0] != _REFERENCE_MAGIC:
        return None
    ndim = int(w[1])
    count = int(np.prod(w[2 : 2 + ndim]))
    return w[2 + ndim : 2 + ndim + count]
