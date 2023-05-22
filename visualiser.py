import pandas as pd
import matplotlib.pyplot as plt
import itertools

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import CustomJS, HoverTool, PanTool, WheelZoomTool, LinearAxis
from bokeh.models.widgets import Button
from bokeh.layouts import column, row


"""
Plot TSNE as an interactive html plot using Bokeh
"""

def make_df(feats_embedding, labels, domain, ids, data_csv, img_paths, preds, name):
    # print(len(feats_embedding))
    # print(feats_embedding)
    # print(len(labels))
    # print(labels)
    # print(len(domain))
    # print(len(ids))
    # print(ids)
    # print(domain)
    # print(len(preds))
    # print(len(img_paths))
    # exit()
    accurate = [1 if x == y else 0 for x, y in zip(labels, preds)]

    samples = {'label': labels,
               'domain': domain,
               'file': ids,
               'img_paths': img_paths,
               'pred': preds,
               'accurate': accurate,
               'x_feats': feats_embedding[:, 0],
               'y_feats': feats_embedding[:, 1]
               }

    sample_df = pd.DataFrame(samples)

    data_csv = data_csv.drop(columns=['label'])
    # data_csv['source_code'] = data_csv['source'].astype('category').cat.codes
    # print(data_csv['source_code'].unique())

    sample_df = pd.merge(sample_df, data_csv, how="inner", on=["file"])
    sample_df.to_csv(f"interactive_plots/{name}_samples.csv")
    print(sample_df.columns)

    # sample_df = pd.DataFrame
    #
    # sample_df['x_feats'] = feats_embedding[:, 0]
    # sample_df['y_feats'] = feats_embedding[:, 1]
    # sample_df['label'] = labels
    # sample_df['domain'] = domain

    vars = ['label', 'domain', 'pred', 'accurate', 'source']
    var_cats = []

    for var in vars:
        sample_df[var] = sample_df[var].astype('category')
        var_cats.append(f'{var}_code')


    sample_df[var_cats] = sample_df[vars].apply(lambda x: x.cat.codes)

    for col in var_cats:
        my_col_string = 'color_' + col
        sample_df[my_col_string] = set_colors(sample_df[col].values, plt.cm.Set1)
    sample_df['color_data'] = set_colors(sample_df.label_code)

    label_dict = {0: 'No Covid', 1: 'Covid'}
    sample_df['label_verbose'] = sample_df['label'].map(label_dict)
    domain_dict = {0: 'Source', 1: 'Target'}
    sample_df['domain_verbose'] = sample_df['domain'].map(domain_dict)
    acc_dict = {0: 'Incorrect', 1: 'Correct'}
    sample_df['accurate_verbose'] = sample_df['accurate'].map(acc_dict)

    return sample_df


def clamp(x):
    return max(0, min(x, 255))


def set_colors(vals_for_color, colors=plt.cm.tab20b):
    min_val = min(vals_for_color);
    max_val = max(vals_for_color)
    vals_for_color_norm = [(float(val) - min_val) / (max_val - min_val) for val in
                           vals_for_color]  # between 0 and 1
    vals_for_color_norm = [val if val < 1 else 0.9999 for val in vals_for_color_norm]

    # colors_unit = [plt.cm.tab20b(val)[:3] for val in vals_for_color_norm]

    colors_unit = [colors(val)[:3] for val in vals_for_color_norm]
    colors_rgb = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors_unit]

    colors_hex = ["#{0:02x}{1:02x}{2:02x}".format(clamp(color_rgb[0]), clamp(color_rgb[1]), clamp(color_rgb[2])) for
                  color_rgb in colors_rgb]

    return colors_hex


def create_bokeh(dann_tsne, labels, domains, ids, data_csv, img_paths, preds, title):
    df = make_df(dann_tsne, labels, domains, ids, data_csv, img_paths, preds, title)

    output_file(f"interactive_plots/{title}.html")
    figure_size = 500

    hover = HoverTool(
        tooltips="""
            <div>
                <span style="font-size: 10px;">Label: @label_verbose</span>
                <br>
                <span style="font-size: 10px;">Domain: @domain_verbose</span>
                <br>
                <span style="font-size: 10px;">Pred: @pred</span>
                <br>
                <span style="font-size: 10px;">Accurate: @accurate_verbose</span>
                <br>
                <span style="font-size: 10px;">Source: @source</span>
                <br>
            </div>
            <div>
                <img
                    src="@img_paths" height="200" width="200"
                    style="float: center;"
                    border="2"
                ></img>
            </div>
            """
    )

    df['color_data'] = set_colors(df.source_code)

    source = ColumnDataSource(data=df.to_dict('list'))
    source2 = ColumnDataSource(data=df.to_dict('list'))

    p = figure(tools=[hover, PanTool(), WheelZoomTool()],
               min_width=figure_size + 500, min_height=figure_size,
               toolbar_location="above", title=title)

    p.circle('x_feats', 'y_feats', fill_color='color_data', legend_field='label', source=source,
             line_color='black', size=10, alpha=0.7)

    p.xaxis.visible = False
    p.yaxis.visible = False
    xaxis = LinearAxis(axis_label="X-embedding")
    yaxis = LinearAxis(axis_label="Y-embedding")
    p.add_layout(xaxis, 'below')
    p.add_layout(yaxis, 'left')
    p.background_fill_color = "#dddddd"
    # p.legend.click_policy = "hide"

    callback = CustomJS(args=dict(source=source, source2=source2, xaxis=xaxis, yaxis=yaxis, legend=p.legend.items[0]),
                        code="""
        var data = source.data;
        var data2 = source2.data;
        data['color_data'] = data2[cb_obj.origin.name];
        legend.label.field = cb_obj.origin.label;
        source.change.emit();
    """)

    toggle1 = Button(label="label", name="color_label_code")
    toggle2 = Button(label="domain", name="color_domain_code")
    toggle3 = Button(label="pred", name="color_pred_code")
    toggle4 = Button(label="accurate", name="color_accurate_code")
    toggle5 = Button(label="source", name="color_source_code")

    toggle1.js_on_click(callback)
    toggle2.js_on_click(callback)
    toggle3.js_on_click(callback)
    toggle4.js_on_click(callback)
    toggle5.js_on_click(callback)

    layout = column(p, row(toggle1, toggle2, toggle3, toggle4, toggle5))
    show(layout)
