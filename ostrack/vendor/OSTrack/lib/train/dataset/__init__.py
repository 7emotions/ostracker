from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .imagenetvid import ImagenetVID
from .got10k_lmdb import Got10k_lmdb
from .lasot_lmdb import Lasot_lmdb
from .imagenetvid_lmdb import ImagenetVID_lmdb
from .tracking_net_lmdb import TrackingNet_lmdb

try:
    from .coco import MSCOCO
    from .coco_seq import MSCOCOSeq
    from .coco_seq_lmdb import MSCOCOSeq_lmdb
except ImportError:
    MSCOCO = None
    MSCOCOSeq = None
    MSCOCOSeq_lmdb = None
