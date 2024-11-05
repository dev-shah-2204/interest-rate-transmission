from flask import Blueprint, render_template, request
from . import bayesnet22 as bayesnet

views = Blueprint("views", __name__)

def check_stationarity(res):
    for item in res:
        if item == "Stationarity":
            for value in res[item]:
                if "Non" in value:
                    return False

    return True


@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # date = str(request.form.get('date'))


        # if date == "2011":
        #     from . import bayesnet22 as bayesnet
        #     date_range = "2011 to 2022"
        #
        # elif date == "2022":
        #     from . import bayesnet24 as bayesnet
        #     date_range = "2022 to 2024"

        date_range = "2011 to 2022"
        adf_result = bayesnet.get_adf_result()
        is_stationary = check_stationarity(adf_result)

        step = request.form.get('step')

        if not is_stationary:
            adf_result_table = adf_result.to_html(classes="table table-striped", index=False)
            correlation_matrix_table = None

        if step == "next":
            bayesnet.get_dag()

            adf_result = bayesnet.get_adf_after_diff()
            is_stationary = check_stationarity(adf_result)
            adf_result_table = adf_result.to_html(classes="table table-striped", index=False)

            correlation_matrix = bayesnet.get_correlation_matrix()
            correlation_matrix_table = correlation_matrix.to_html(classes="table table-striped", index=False)

        return render_template('index.html', is_stationary=is_stationary, show_res=True, adf_table=adf_result_table, correlation_matrix_table=correlation_matrix_table, date_range=date_range)

    return render_template('index.html', show_res=False)


@views.route('/test', methods=['GET'])
def test():
    return render_template('test2.html')

@views.route('/about', methods=['GET'])
def about():
    return render_template('about.html')
