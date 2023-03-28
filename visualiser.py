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

def make_df(feats_embedding, labels, domain, img_paths, name):
    labels = list(itertools.chain.from_iterable(labels))

    samples = {'label': labels,
               'domain': domain.cpu().numpy(),
               'img_paths': img_paths,
               'x_feats': feats_embedding[:, 0],
               'y_feats': feats_embedding[:, 1]
               }

    sample_df = pd.DataFrame(samples)
    # sample_df = pd.DataFrame
    #
    # sample_df['x_feats'] = feats_embedding[:, 0]
    # sample_df['y_feats'] = feats_embedding[:, 1]
    # sample_df['label'] = labels
    # sample_df['domain'] = domain

    vars = ['label', 'domain']
    var_cats = []

    for var in vars:
        sample_df[var] = sample_df[var].astype('category')
        var_cats.append(f'{var}_code')

    sample_df[var_cats] = sample_df[vars].apply(lambda x: x.cat.codes)

    for col in var_cats:
        my_col_string = 'color_' + col
        sample_df[my_col_string] = set_colors(sample_df[col].values, plt.cm.Set1)

    sample_df['color_data'] = set_colors(sample_df.label_code)
    sample_df.to_csv(f"{name}_samples.csv")
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


def create_bokeh(dann_tsne, labels, domains, img_paths, title):
    df = make_df(dann_tsne, labels, domains, img_paths, title)

    print(df.head())

    output_file(f"{title}.html")
    figure_size = 500


    hover = HoverTool(
        tooltips="""
            <div>
                <span style="font-size: 10px;">Label: @label</span>
                <br>
                <span style="font-size: 10px;">domain: @domain</span>
                <br>
                <span style="font-size: 10px;">domain: @domain</span>
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

    df['color_data'] = set_colors(df.label)

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
    p.legend.click_policy = "hide"

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

    toggle1.js_on_click(callback)
    toggle2.js_on_click(callback)

    layout = column(p, row(toggle1, toggle2))
    show(layout)
