
# delete
def sum_cluster(clusterer, doc=True, n=10):
    matrix = clusterer.rows_ if doc else clusterer.columns_
    title = "Docs" if doc else "Topics"
    clusters = range(len(matrix))
    doc_indices = np.arange(len(matrix[0]))
    for c in clusters:
        # bool indicating cluster membership
        doc_where = matrix[c]
        c_indices = doc_indices[doc_where]
        try:
            c_indices = np.random.choice(c_indices, size=n, replace=False)
        except ValueError:
            pass # tried to sample w/ 'size' > len
        to_print = f"*** {title} in Cluster {c} ***\n\n"
        for i,ind in enumerate(c_indices):
            if doc:
                thing = tweets[ind]
            else:
                thing = list(topic.top_topic_terms(cv.get_feature_names(), top_n=6))[ind]
            to_print += f"{i}. {thing}\n"
        print(to_print)


# Recursively cluster a single cluster
def recurse_biclustering(clusterer, data, coords):
    rows = np.where(np.equal(clusterer.row_labels_, coords[0]))[0]
    cols = np.where(np.equal(clusterer.column_labels_, coords[1]))[0]
    data = data.copy()
    data = data[rows][:,cols]
    new_clusterer = SpectralBiclustering()
    new_clusterer.set_params(**clusterer.get_params())
    new_clusterer.fit(data)
    return new_clusterer, data

# Interactively
def inspect(coords):
    global binary_spec, doc_term_bin, cv
    b, d = recurse_biclustering(binary_spec, doc_term_bin, coords)
    _,ax = plt.subplots()
    draw_image_matrix(b, d, cv, ax, True)
    fname = "visualization/inspect.png"
    print("saving: "+fname)
    plt.savefig(fname, dpi=500)
    plt.close('all')


# re-create the 'lognormalization' from Kluger et al
# log(1+x) and then doubly center
def lognormalize(data):
    data = data.copy()
    data = np.log1p(data)
    rmean = np.mean(data, axis=1)
    cmean = np.mean(data, axis=0)
    rmean = np.broadcast_to(rmean, data.shape)
    cmean = np.broadcast_to(cmean, data.shape)
    mean = np.broadcast_to(np.mean(data), data.shape)
    data = data - cmean - rmean + mean
    return data
    

# works for bi- or co-clustering!
# but hardly shows up for sparse data
def draw_matrix(clusterer, data, lognormalize=False):
    if lognormalize:
        data = lognormalize(data)
    # sort
    data = data[np.argsort(clusterer.row_labels_)]
    data = data[:,np.argsort(clusterer.column_labels_)]
    try:
        plt.matshow(data, cmap=plt.cm.Blues)
    except ValueError:
        plt.matshow(data.todense(), cmap=plt.cm.Blues)


# best to call w/ axis from plt.subplots(tight_layout=True)
def draw_image_matrix(clusterer, data, cv=None, ax=plt, lognormalize=False, percentile=False):
    image, counts = get_image_matrix(clusterer, data, lognormalize, percentile)
    #image = image.transpose()
    ax.matshow(image, cmap=plt.cm.Blues)
    # set tick labels
    yticks = np.array(range(image.shape[0]))
    ylabels = list(map(lambda x: "\n".join(x),
                       get_handles_by_cluster(clusterer, data, 3, descriptions=True)))
    try:
        # artist-style
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, size=3)
    except AttributeError:
        # scripting-style
        ax.yticks(yticks, labels=[])
    xticks = np.array(range(image.shape[1]))
    if cv:
        xlabels = list(map(lambda x: "\n".join(x), get_terms_by_cluster(clusterer, data, cv, 3)))
    else:
        xlabels=None
    try:
        # scripting-style
        ax.xticks(ticks=xticks, labels=xlabels, size=4)
    except AttributeError:
        # artist-style
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=90, size=4)
    # annotate w/ counts
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            # don't transpose 'counts' b/c plt.matshow orients axes funny
            ax.annotate(counts[i,j], (j,i), size=3, ha='center')
    


def get_image_matrix(clusterer, data, lognormalize=False, percentile=False):
    if lognormalize:
        data = lognormalize(data)
    clusters = [pd.unique(clusterer.row_labels_), pd.unique(clusterer.column_labels_)]
    dim = list(map(lambda x: len(x), clusters))
    image = np.zeros(shape=dim)
    counts = np.full(shape=dim, fill_value='', dtype=object)
    for i in clusters[0]:
        for j in clusters[1]:
            submat = get_bicluster_submatrix(clusterer, i, j, data)
            if percentile is False:
                image[i,j] = np.mean(submat)
            else:
                image[i,j] = np.percentile(submat, percentile)
            counts[i,j] = f"{submat.shape[0]}x{submat.shape[1]}"
    return image, counts

def get_bicluster_submatrix(clusterer, i, j, data):
    rows = np.where(np.equal(clusterer.row_labels_, i))[0]
    columns = np.where(np.equal(clusterer.column_labels_, j))[0]
    return data[rows][:,columns]
    

    

def print_by_cluster(clusterer, doc_term, cv, n_terms, words=True, descriptions=False):
    global rec_handles, tw_df
    if words and descriptions:
        print("'descriptions=true' only pertains to handles")
        return
    if words:
        top_terms = get_terms_by_cluster(clusterer, doc_term, cv, n_terms)
    else:
        top_terms = get_handles_by_cluster(clusterer, doc_term, n_terms)
    for i in range(len(top_terms)):
        print(i)
        if descriptions:
            # loop thru handles
            for h in top_terms[i]:
                # print in full
                desc = get_handle_description(h, tw_df, 1000)
                desc = "; ".join(desc.split("\n"))
                print(f"   {h} -- {desc}")
        else:
            # print compact list-style
            print(f"\t{top_terms[i][:n_terms]}")

def get_terms_by_cluster(clusterer, doc_term, cv, n_terms=5):
    # get cluster indices
    col_clusters = pd.unique(clusterer.column_labels_)
    col_clusters = np.sort(col_clusters)
    # get term frequencies
    freq = np.sum(doc_term, axis=0)
    # grr matrices
    if len(freq.shape)>1:
        freq = np.array(freq)
        freq = freq[0]
    # get int->string vocabulary
    words = cv.get_feature_names()
    top_terms = []
    for c in col_clusters:
        # get term indices
        term_inds = np.where(np.equal(clusterer.column_labels_, c))[0]
        # get frequencies
        term_freqs = freq[term_inds]
        # get top frequencies
        top_inds = term_inds[np.argsort(term_freqs)[-1*n_terms:]]
        top_cluster_terms = [words[i] for i in top_inds]
        top_terms.append(top_cluster_terms)
    return top_terms


def get_handles_by_cluster(clusterer, doc_term, n, descriptions=False):
    global rec_handles, tw_df
    # get cluster indices
    row_clusters = pd.unique(clusterer.row_labels_)
    row_clusters = np.sort(row_clusters)
    # get usage frequencies
    freq = np.sum(doc_term, axis=1)
    # grr matrices
    if len(freq.shape)>1:
        freq = np.array(freq)
        freq = freq[:,0]
    # get int->string vocabulary (here it's rec_handles)
    top_handles = []
    for c in row_clusters:
        # get term indices
        handle_inds = np.where(np.equal(clusterer.row_labels_, c))[0]
        # get frequencies
        handle_freqs = freq[handle_inds]
        # get top frequencies
        top_inds = handle_inds[np.argsort(handle_freqs)[-1*n:]]
        if descriptions:
            top_cluster_handles = [get_handle_description(rec_handles[i], tw_df)
                                   for i in top_inds]
        else:
            top_cluster_handles = [rec_handles[i] for i in top_inds]
        top_handles.append(top_cluster_handles)
    return top_handles
    
def get_handle_description(handle, tw_df, nchar=24):
    description = tw_df.query('handle == @handle').iloc[0,:].description
    if pd.isna(description):
        description = str(description)
    else:
        description = description[:nchar]
    return description
