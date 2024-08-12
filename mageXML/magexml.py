from PIL import Image
from trp import TextLine, PageXML ## need these from trp
import numpy as np
import os
import json
import traceback


## read xml file
def parse_xml(xml_file_name: str) -> PageXML:
    return PageXML(xml_file_name)



## find edges of columns
def find_edges(page: PageXML, threshold=0.2, gap=0.03) -> list[int]:
    """
    returns list of possible edges of columns, does not work well with overlapping columns

    Parameters
    ----------
    page : PageXML
        page of which we want the find the columns

    threshold : float
        width of the page (as a ratio) that we use as the window to detect the edge of a column

    gap : float
        essentially how lenient the function is to lines of a column being slightly offset from each other. smaller value is more strict 
    
    Returns
    -------
    list[int]
        list of the x-values of the edges of each column. will always return an even number of edges
    """

    x, y = page.get_image_dims()
    x = int(x)
    y = int(y)

    baselines = page.get_all_lines()

    edges = []

    ## get min/max x of each line
    xtrema = []
    for line in baselines:
        validate_textlines(line)
        min_x = np.min(line.bl_pts, axis=0)[0]
        max_x = np.max(line.bl_pts, axis=0)[0]
        xtrema.append([min_x, max_x])

    ## start at left edge of page
    ## find baselines with minx within 0.2 of page
    ## if minx are within 0.03 of 0.2 threshold, search within 0.03 of largest minx
    ## add it to those found
    ## find max maxx of those baselines, that is an edge
    ## start again at 0.4 of page, looking for minx
    l = 0
    r = threshold
    while r < 1 + gap:
        rs = r * x
        ls = l * x
        ## finding minx
        minx_lines = [ t for t in xtrema if (t[0] <= rs and t[0] >= ls) ]

        if not minx_lines:
            l = r
            r += threshold
            continue

        ## check close to threshold, add to those found
        too_close = np.abs(np.max(minx_lines, axis=0)[0] - rs) <= 0.03 * x
        if too_close:
            new_r = np.max(minx_lines, axis=0)[0] + 0.03 * x
            for t in xtrema:
                if t[0] > rs and t[0] <= new_r: 
                    minx_lines.append(t)
        
        ## add to edges
        edges.append(np.min(minx_lines, axis=0)[0])
        edges.append(np.max(minx_lines, axis=0)[1])

        ## iterate
        l = edges[-1] / x
        r = edges[-1] / x + threshold

    return edges



## determine column a line is in
def determine_column(edges: list[int], line: TextLine) -> int:
    """
    returns the index of the column that the textline is in

    Parameters
    ----------
    edges : list[int]
        x-values of the edges of all the columns

    line : TextLine
        the line of which we want to determine the column
    
    Returns
    -------
    int
        column's index, -1 if the line is not fully contained in one of the columns
    """

    ## if only one column
    if len(edges) == 2:
        return 0
    
    validate_textlines(line)
    
    min_x = np.min(line.bl_pts, axis=0)[0]
    max_x = np.max(line.bl_pts, axis=0)[0]
    
    ## check where the line starts in comparison to edges
    for i in range(0, len(edges), 2):
        if min_x >= edges[i] and max_x <= edges[i+1]:
            ## line is within column
            return int(i / 2)
    ## if the line didn't fit in any column:
    return -1




## validate textlines, make sure they are the right format
def validate_textlines(line: TextLine) -> None:
    """
    ensure that baselines are properly formatted

    Parameters
    ----------
    line : TextLine
        the line we would like to check

    Returns
    -------
    None
    """

    assert type(line.bl_pts) == np.ndarray
    
    if line.bl_pts.shape[0] <= 1:
        raise Exception("TextLine baseline array contains less than two points")
    
    if np.min(line.bl_pts) < 0:
        raise Exception("TextLine baseline has negative coordinates") 
    
    if len(line.bl_pts.shape) != 2 or line.bl_pts.shape[1] != 2:
        raise Exception(f"TextLine baseline of shape {line.bl_pts.shape} is not the correct dimensions for an array of coordinates") 


## group all lines in a page by their columns
def group_by_column(page: PageXML) -> dict[int , list[TextLine]]:
    """
    groups all textlines in page into their respective columns, returns dict of the groupings

    Parameters
    ----------
    page : PageXML
        the page of which we are group its lines

    Returns
    -------
    dict[int, list[TextLine]]
        the keys of the dictionary are integers from -1 to the number of columns minus 1, the values are the lists of TextLines that correspond to the column represented by the key
    """

    ## BEWARE, the output of this function is a dictionary that has
    ## integer keys, some of which may be -1
    ## do not mistake this for list indexing, where -1 means the
    ## last element

    edges = find_edges(page)
    
    num_cols = int(len(edges) / 2)

    ## if only one column
    if num_cols == 1:
        return {0: page.get_all_lines()}

    columns = { i : [] for i in range(-1, num_cols)}

    for line in page.get_all_lines():
        col = determine_column(edges, line)
        columns[col].append(line)

    return columns



## create bounding box for textlines, BASED OFF BASELINES
## in other words, the first line will be chopped off
## this is dealt with later in the determine_slices function, where the
## top y value of the box is changed to fit that first line
def bounding_box(baselines: list[TextLine]) -> list[list[int]]:
    """
    returns a tuple of two points corresponding to the top left and bottom right of the bounding box
    
    Parameters
    ----------
    baselines : list[Textlines]
        list of the textlines
    
    Returns
    -------
    tuple
        tuple of the bounding box's top right and bottom left coordinate
    """

    ## need to check how to get the top of first line

    min_x, min_y = np.inf, np.inf
    max_x, max_y = 0, 0

    for line in baselines:
        validate_textlines(line)
        for point in line.bl_pts:
            x = point[0]
            y = point[1]
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y

    return [min_x, min_y], [max_x, max_y]



## crop the image
def slice_img(page: PageXML, coords: list[list], image_dir="") -> Image:
    """
    returns an Image cropped to the text region of the page marked by the coords

     Parameters
    ----------
    page : PageXML
        the PageXML object of the page we want to crop

    coords : list[list]
        a list of two points, the first is the top-left point of the region and the second is the bottom-right corner
    
    Returns
    -------
    Image
        the cropped Image
    """

    image_name, x, y = page.get_image_data()
    x = int(x)
    y = int(y)

    top, right, bottom, left = coords[0][1], coords[1][0], coords[1][1], coords[0][0]
    
    if image_dir:
        image_name = image_dir + "/" + image_name

    im = Image.open(image_name)

    new_im = im.crop((left, top, right, bottom))

    return new_im



## get text from column
def slice_from_col(lines: list[TextLine], min_height=0, max_height=0) -> list[TextLine]:
    """
    returns lines of a page between thresholds min_height and max_height, from the provided lines

     Parameters
    ----------
    lines : list[TextLine]
        the lines of the column we want to extract from

    min_height : int
        (optional) minimum threshold over which to consider baselines. 

    max_height : int
        (optional) maximum threshold under which to consider baselines. using 0 returns all baselines
    
    Returns
    -------
    list[Textlines]
        the lines of the slice
    """

    sliced_lines = []

    if max_height == 0:
        max_height = np.inf

    for line in lines:
        validate_textlines(line)
        in_bounds = np.min(line.bl_pts, axis=0)[1] >= min_height and np.max(line.bl_pts, axis=0)[1] < max_height
        if in_bounds:
            sliced_lines.append(line)

    return sliced_lines



## determine image slices and ground truth of each slice
def decide_slices(page: PageXML, pred_length=140) -> list[dict]:
    """
    returns a list of dicts, each dict containing the top left point coordinates, bottom right point coordinates, and ground_truth of a slice

     Parameters
    ----------
    page : PageXML
        the PageXML object of the page we want to slice

    pred_length : int
        maximum length of the prediction by Donut, each image slice will have this many characters or less, if possible
    
    Returns
    -------
    list[dict]
        a list of dicts containing top left point coordinates, bottom right point coordinates, and ground_truth of each slice
    """

    ## get image dims
    x, y = page.get_image_dims()

    y = int(y)

    slices = []


    ## greedy approach w/ column knowledge

    ## get column edges
    edges = find_edges(page)
    ## get number of columns
    num_cols = len(edges) / 2

    columns = group_by_column(page)

    ## for each column, go down the page
    for col, lines in columns.items():
        if len(lines) == 0:
            continue
        ## bottom is how far we have sliced so far down the page
        bottom = 0

        ## growth rate: about how large each slice should be as a percentage of page height
        gr = 0.1

        while bottom < y:
            ## potential bottom of next slice
            new_bottom = bottom + np.round(gr * y)

            if new_bottom - bottom < 2:
                raise Exception("growth rate is too small, and no text slices are being produced")

            new_text = slice_from_col(lines, bottom, new_bottom)
            
            gt = "".join([ line.txt for line in new_text if line.txt is not None])
            new_text_len = len(gt)


            ## one line is longer than the pred_length and we should skip this line
            if len(new_text) == 1 and new_text_len > pred_length:
                coords = bounding_box(new_text)
                bottom = coords[1][1] + 1
            elif new_text_len == 0:
                if new_bottom > y:
                    break
                gr *= 1.2 + np.random.uniform(-0.1, 0.05)
            elif new_text_len > pred_length:
                gr *= pred_length / new_text_len
            elif new_text_len < 0.1 * pred_length:
                gr *= (0.1 * pred_length) / new_text_len
                coords = bounding_box(new_text)
                if -1 in columns and col != -1:
                    ## if there are lines that don't match any column, check
                    ## if they are closer than bottom, and use their baseline instead
                    nearest_bl = bottom
                    for line in columns[-1]:
                        if line.txt == "":
                            continue
                        max_y = np.max(line.bl_pts, axis=0)[1]
                        if max_y > nearest_bl and max_y < coords[1][1] - 0.01 * y:
                            nearest_bl = max_y
                    bottom = nearest_bl

                coords[0][1] = bottom
                real_bottom = coords[1][1] + 1

                ## add padding
                coords[1][1] += int(0.005 * y)

                slices.append({
                    "coords": coords,
                    "ground_truth": gt
                })
                bottom = real_bottom
            else:
                ## goldilocks zone for the slice
                coords = bounding_box(new_text)
                if -1 in columns and col != -1:
                    ## if there are lines that don't match any column, check
                    ## if they are closer than bottom, and use their baseline instead
                    nearest_bl = bottom
                    for line in columns[-1]:
                        if line.txt == "":
                            continue
                        max_y = np.max(line.bl_pts, axis=0)[1]
                        if max_y > nearest_bl and max_y < coords[1][1] - 0.01 * y:
                            nearest_bl = max_y
                    bottom = nearest_bl
                coords[0][1] = bottom
                real_bottom = coords[1][1] + 1

                ## add padding
                coords[1][1] += int(0.005 * y)

                slices.append({
                    "coords": coords,
                    "ground_truth": gt
                })
                bottom = real_bottom

    return slices



## write metadata to file
def create_metadata(page: PageXML, pred_length=140) -> None:
    """
    saves image slices as new images and writes metadata to file in Donut format

    Parameters
    ----------
    page : PageXML
        the PageXML object of the page we want to create training data from

    pred_length : int
        maximum length of the prediction by Donut, each image slice will have this many characters or less, if possible
    
    Returns
    -------
    None
    """

    page_name = page.get_image_data()[0].split('.')[0]
    slices = decide_slices(page, pred_length)

    splits = [0.8, 0.1, 0.1]
    sets = ['train', 'validation', 'test']

    ## make set folders
    for s in sets:
        ## Check if the directory already exists
        if not os.path.exists('dataset/' + s):
            ## Create the directory
            os.makedirs('dataset/' + s)

    for i, s in enumerate(slices):
        ## crop
        im = slice_img(page, s['coords'], 'raw_images')

        ## decide which set
        assigned_set = sets[np.random.choice(3, p=splits)]

        im_name = f'dataset/{assigned_set}/{page_name}_{i}.jpg'

        ## save image
        im.save(im_name)

        d = {
            'file_name': im_name.split('/')[2],
            'ground_truth': f"{{\"gt_parse\": {{\"text_sequence\": \"{s['ground_truth']}\" }} }}"
        }

        ## write metadata
        with open(f"dataset/{assigned_set}/metadata.jsonl", 'a') as f:
            json.dump(d,f)
            f.write('\n')



## take in all xml in folder and make the training data
def create_dataset(xml_dir: str, verbose=True) -> None:
    """
    create the full dataset for donut, based off the xml files in the xml_dir

    images for the pages should be in a folder called 'raw_images' that is in the same parent directory as this python file

    Parameters
    ----------
    xml_dir : str
        the directory of the Transkribus xml files for the pages we want to convert to donut training
    
    verbose: bool
        whether or not to print the name of the file being processed

    Returns
    -------
    None
    """

    dir = os.fsencode(xml_dir)
    
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".xml"): 
            try:
                page = parse_xml('pages/' + filename)
                create_metadata(page)
            except KeyboardInterrupt:
                ## keyboard interrupt will skip a file that is taking too long
                ## it will not stop the program!
                print('Interrupted!')
                print(f'    Error with file: {filename}')
                traceback.print_exc()
            except Exception as e:
                print(f"Error with file: {filename}")
                print(f'    {e}')
                traceback.print_exc()
            if verbose:
                print(f'Processed page: {filename}')
        else:
            continue




###### RUNNING THE CODE, CHANGE THIS LINE IF NECESSARY
create_dataset('pages')
