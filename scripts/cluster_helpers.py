
#
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


# works for bi- or co-clustering!
# but hardly shows up for sparse data
def draw_matrix(clusterer, data):
    # sort
    data = data[np.argsort(clusterer.row_labels_)]
    data = data[:,np.argsort(clusterer.column_labels_)]
    data = np.log(1 + data.todense())
    try:
        plt.matshow(data, cmap=plt.cm.Blues)
    except ValueError:
        plt.matshow(data.todense(), cmap=plt.cm.Blues)


# best to call w/ axis from plt.subplots(tight_layout=True)
def draw_image_matrix(clusterer, data, cv=None, ax=plt):
    image, counts = get_image_matrix(clusterer, data)
    image = image.transpose()
    ax.matshow(image, cmap=plt.cm.Blues)
    xticks = np.array(range(image.shape[1]))
    try:
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
    except AttributeError:
        ax.xticks(xticks, labels=[])
    yticks = np.array(range(image.shape[0]))
    if cv:
        labels = list(map(lambda x: "\n".join(x), get_terms_by_cluster(clusterer, data, cv, 3)))
    else:
        labels=None
    try:
        ax.yticks(ticks=yticks, labels=labels, size='xx-small')
    except AttributeError:
        ax.set_yticks(yticks)
        ax.set_yticklabels(labels, size='xx-small')
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            # don't transpose 'counts' b/c plt.matshow orients axes funny
            ax.annotate(counts[i,j], (i,j), size=3, ha='center')
    


def get_image_matrix(clusterer, data):
    clusters = [pd.unique(clusterer.row_labels_), pd.unique(clusterer.column_labels_)]
    dim = list(map(lambda x: len(x), clusters))
    image = np.zeros(shape=dim)
    counts = np.full(shape=dim, fill_value='', dtype=object)
    for i in clusters[0]:
        for j in clusters[1]:
            submat = get_bicluster_submatrix(clusterer, i, j, data)
            m = np.mean(submat)
            image[i,j] = np.log(1+m)
            # order wonky b/c ptl.matshow orients wonky
            counts[i,j] = f"{submat.shape[1]}x{submat.shape[0]}"
    return image, counts

def get_bicluster_submatrix(clusterer, i, j, data):
    rows = np.where(np.equal(clusterer.row_labels_, i))[0]
    columns = np.where(np.equal(clusterer.column_labels_, j))[0]
    return data[rows][:,columns]
    

    

def print_terms_by_cluster(clusterer, doc_term, cv, n_terms):
    top_terms = get_terms_by_cluster(clusterer, doc_term, cv, n_terms)
    for i in range(len(top_terms)):
        print(i)
        for t in top_terms[i]:
            print(f"\t{t}")

def get_terms_by_cluster(clusterer, doc_term, cv, n_terms=5):
    # get cluster indices
    col_clusters = pd.unique(clusterer.column_labels_)
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
