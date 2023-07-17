import gradio as gr
import datetime
from inference import get_update_data, predict_price
from preprocess import get_technical_indicators



# dataset = get_update_data()

year_list = [str(i) for i in range(2000, 2024)]

month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_dict = {month_list[i]: i + 1 for i in range(len(month_list))}

month30_list = ['April', 'June', 'September', 'November']
month31_list = ['January', 'March', 'May', 'July', 'August', 'October', 'December']

day_list = [str(i) for i in range(1, 32)]
model_list = ['Catboost', 'LightGBM', "XGBoost", "LSTM", "GRU"]

interval_list = ['1 day', '1 week', '1 month']

def change_month(year, month):
    if month in month30_list:
        return gr.update(choices=day_list[:-1], value=day_list[0])
    elif month in month31_list:
        return gr.update(choices=day_list, value=day_list[0])
    else:
        if int(year)%4 == 0:
            return gr.update(choices=day_list[:-2], value=day_list[0])
        else:
            return gr.update(choices=day_list[:-3], value=day_list[0])


def predict(sd, sm, sy, ed, em, ey, m, interval):

    if interval == '1 day':
        interval = 1
    elif interval == '1 week':
        interval = 7
    elif interval == '1 month':
        interval = 30

    start_date = datetime.date(day=int(sd), month=month_dict[sm], year=int(sy)).strftime("%d/%m/%Y")
    end_date = datetime.date(day=int(ed), month=month_dict[em], year=int(ey)).strftime("%d/%m/%Y")
    print(start_date, end_date)
    dataset = get_update_data(start=start_date, end=end_date)

    data = predict_price(m, dataset, interval)

    return gr.LinePlot.update(
        value=data,
        x="date",
        y="close",
        # color="symbol",
        # color_legend_position="bottom",
        title="Stock Prices",
        tooltip=["date", "close"],
        height=500,
        width=1000,
    ) 

def update_data(sd, sm, sy, ed, em, ey):

    start_date = datetime.date(day=int(sd), month=month_dict[sm], year=int(sy)).strftime("%d/%m/%Y")
    end_date = datetime.date(day=int(ed), month=month_dict[em], year=int(ey)).strftime("%d/%m/%Y")

    dataset = get_update_data(start=start_date, end=end_date)

    dataset = get_technical_indicators(dataset)

    return gr.LinePlot.update(
        value=dataset,
        x="date",
        y="value",
        color="type",
        color_legend_position="bottom",
        title="Stock Prices",
        tooltip=["date", "value", "type"],
        height=500,
        width=1000,
    ) 


def line_plot_fn(start_date='01/01/2000', end_date='01/01/2023'):

    dataset = get_update_data(start=start_date, end=end_date)
    return gr.LinePlot.update(
        value=dataset,
        x="date",
        y="close",
        # color="symbol",
        # color_legend_position="bottom",
        title="Stock Prices",
        tooltip=["date", "close"],
        height=500,
        width=1000,
    )


with gr.Blocks(title="Stock Price Prediction") as app:
    with gr.Row():
        start_year = gr.Dropdown(choices=year_list, value=year_list[0], label="Start Year")
        end_year = gr.Dropdown(choices=year_list, value=year_list[-1], label="End Year")
    with gr.Row():
        start_month = gr.Dropdown(choices=month_list, value=month_list[0], label="Start Month")
        end_month = gr.Dropdown(choices=month_list, value=month_list[0], label="End Month")
    with gr.Row():
        start_day = gr.Dropdown(choices=day_list, value=day_list[0], label="Start Month")
        end_day = gr.Dropdown(choices=day_list, value=day_list[0], label="End Month")
    with gr.Row():
        model = gr.Dropdown(choices=model_list, value=model_list[0], label="Model")
        interval = gr.Dropdown(choices=interval_list, value=interval_list[0], label="Model")
    with gr.Row():
        update_data_btn = gr.Button(value="Update Data")
        submit_btn = gr.Button(value="Submit")

    with gr.Row():
        plot = gr.LinePlot()
    # dataset.change(line_plot_fn, inputs=dataset, outputs=plot)

    start_month.change(fn=change_month, inputs=[start_year, start_month], outputs=start_day)
    end_month.change(fn=change_month, inputs=[end_year, end_month], outputs=end_day)

    app.load(fn=line_plot_fn, outputs=plot)
    submit_btn.click(fn=predict, inputs=[start_day, start_month, start_year, end_day, end_month, end_year, model, interval], outputs=plot)
    update_data_btn.click(fn=update_data, inputs=[start_day, start_month, start_year, end_day, end_month, end_year], outputs=plot)


app.launch(server_port=8080)
