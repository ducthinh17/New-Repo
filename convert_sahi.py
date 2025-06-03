def convert_sahi_results_to_bounding_boxes(sahi_results):
    """
    Convert SAHI prediction results to bounding box format.
    """
    bounding_boxes = []
    cords_arr = []
    class_ids = []

    for result in sahi_results:
        class_id = result.category.name  # Correctly extract the class name
        bbox = result.bbox.to_xyxy()  # Extract bounding box coordinates
        score = result.score.value  # Extract confidence score

        cords_arr.append([int(coord) for coord in bbox])
        bounding_boxes.append({
            "class_id": class_id,
            "cords": [int(coord) for coord in bbox],
            "percentage_conf": f"{score * 100:.0f}"
        })
        class_ids.append(class_id)

    return bounding_boxes, len(bounding_boxes), cords_arr, class_ids
