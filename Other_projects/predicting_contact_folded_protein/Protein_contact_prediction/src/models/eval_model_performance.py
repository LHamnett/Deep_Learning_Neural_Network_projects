from sklearn.metrics import matthews_corrcoef
import pandas as pd
import yaml
import os

def generate_baseline_metric_per_row(df, y_true_col, y_pred_col, bin_threshold=0.5):
    """
    Compute Matthews Correlation Coefficient (MCC) for each row of a DataFrame,
    where each row contains a contact map (2D array) in both true and predicted columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns with ground truth and predicted contact maps.
    y_true_col : str
        Column name containing the ground-truth binary contact maps.
    y_pred_col : str
        Column name containing the predicted probability contact maps.
    bin_threshold : float, optional
        Threshold to binarise predicted probabilities.

    Returns
    -------
    pd.Series
        MCC per row.
    float
        Average MCC across all rows.
    """
    mcc_scores = []

    for _, row in df.iterrows():
        y_true = np.array(row[y_true_col]).ravel()
        y_pred = (np.array(row[y_pred_col]) >= bin_threshold).astype(int).ravel()

        # Handle potential all-zero cases safely
        if len(np.unique(y_true)) == 1:
            mcc = np.nan  # undefined when all true values are the same
        else:
            mcc = matthews_corrcoef(y_true, y_pred)

        mcc_scores.append(mcc)

    df["mcc"] = mcc_scores
    avg_mcc = np.nanmean(mcc_scores)

    return df["mcc"], avg_mcc

mismatched_shape_df = merged_df[merged_df['gt_contact_map_shapes'] != merged_df['esm2_cont_preds_shapes']]
merged_df_bad_rows_dropped = merged_df.drop(mismatched_shape_df.index)

merged_df_bad_rows_dropped["mcc"] , avg_mcc = generate_baseline_metric_per_row(
    merged_df_bad_rows_dropped,
    y_true_col="contact_map",
    y_pred_col="esm2_contact_map_preds",
    bin_threshold=0.5
)
