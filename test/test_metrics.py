"""Testing for the ``defect_detection metrics``."""
import torch
import numpy as np
import defect_detection

def test_fB_worst_case():
    """Test worst case of fbeta."""
    assert defect_detection.performance_metrics.fB_score(0.0, 0.0) == 0.0

def test_fB_best_case():
    """Test best case of fbeta."""
    assert defect_detection.performance_metrics.fB_score(1.0, 1.0) == 1.0

def test_fB_coin_flip():
    """Test coin flip fbeta."""
    assert np.abs(defect_detection.performance_metrics.fB_score(0.5, 1.0)
                  - 0.667) < 0.01

def test_fB_beta():
    """Test coin flip fbeta with a different beta."""
    assert np.abs(defect_detection.performance_metrics.fB_score(0.5, 1.0, 0.5)
                  - 0.555) < 0.01

def test_iou_report():
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[48:, :] = 1
    true = true.flatten()
    pred = pred.flatten()
    _, df, _ = defect_detection.performance_metrics.iou_report(true, pred, names=['x', 'y'])
    df = df.T
    #df = df.rename(columns=df.iloc[0])
    #df = df.drop(index='Measure', axis=0)
    assert df['IoU'].mean() == 0.0
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[:48, :] = 1
    true = true.flatten()
    pred = pred.flatten()
    _, df, _ = defect_detection.performance_metrics.iou_report(true, pred, names=['x', 'y'])
    df = df.T
    #df = df.rename(columns=df.iloc[0])
    #df = df.drop(index='Measure', axis=0)
    assert df['IoU'].mean() == 1.0
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[:, 48:] = 1
    true = true.flatten()
    pred = pred.flatten()
    _, df, _ = defect_detection.performance_metrics.iou_report(true, pred, names=['x', 'y'])
    df = df.T
    #df = df.rename(columns=df.iloc[0])
    #df = df.drop(index='Measure', axis=0)
    assert np.abs(df['IoU'].mean() - 0.333) < 0.01
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[:, 48:] = 1
    true = true.flatten()
    pred = pred.flatten()
    _, df, _ = defect_detection.performance_metrics.iou_report(true, pred, weights='equal')
    df = df.T
    #df = df.rename(columns=df.iloc[0])
    #df = df.drop(index='Measure', axis=0)
    assert np.abs(df['IoU'].mean() - 0.333) < 0.01

def test_pixel_proportion():
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[48:, :] = 1
    true = true.flatten()
    pred = pred.flatten()
    df = defect_detection.performance_metrics.pixel_proportion(true, pred)
    assert df['Proportion'].mean() == 0.5

def test_mod_report():
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[:, :48] = 1
    true = true.flatten()
    pred = pred.flatten()
    (_, fscoreA, _) = defect_detection.performance_metrics.modified_class_report(true, pred)
    assert np.all(fscoreA == 0.5)
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[:, :48] = 1
    true = true.flatten()
    pred = pred.flatten()
    (_, fscoreA, _) = defect_detection.performance_metrics.modified_class_report(true, pred, weights='equal')
    assert np.all(fscoreA == 0.5)
    true = np.zeros((96, 96))
    true[:48, :] = 1
    pred = np.zeros((96, 96))
    pred[:, :48] = 1
    true = true.flatten()
    pred = pred.flatten()
    (_, fscoreA, _) = defect_detection.performance_metrics.modified_class_report(true, pred, weights='other')
    assert np.all(fscoreA == 0.5)

if __name__ == "__main__":
    test_mod_report()
