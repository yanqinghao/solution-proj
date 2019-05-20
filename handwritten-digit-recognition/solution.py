from mlstudiosdk.solution_gallery.solution.SolutionBase import SolutionBase

class Solution(SolutionBase):

    def __init__(self):
        super().__init__()
        self.model()

    def model(self):
        # reader = self.myscheme.new_node('reader.Reader')
        # reader.set_title('input')
        # cnn = self.myscheme.new_node('cnn.CNN')
        # cnn.set_title('algorithm')
        # writer = self.myscheme.new_node('writer.Writer')
        # writer.set_title('output')
        #
        # self.myscheme.new_link(reader,'Data',cnn,'Data_IN')
        # self.myscheme.new_link(cnn,'Data_OUT',writer,'Data')

        reader1 = self.myscheme.new_node("mlstudiosdk.modules.components.io.reader.Reader")
        reader1.set_title("train input")
        reader2 = self.myscheme.new_node("mlstudiosdk.modules.components.io.reader.Reader")
        reader2.set_title("test input")
        # impute = self.myscheme.new_node("mlstudiosdk.modules.components.preprocess.impute.Impute")
        # impute.set_title("impute")
        # impute.set_default_method(2)
        cnn = self.myscheme.new_node('cnn1.CNN')
        cnn.set_title('algorithm')
        outputwriter = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.Writer")
        outputwriter.set_title("output")
        eva_visualization = self.myscheme.new_node(
            "mlstudiosdk.modules.components.visualization.evaluation_matrix.Evaluation")
        eva_visualization.set_title("evaluation_visualization")
        evaluation_writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.JsonWriter")
        evaluation_writer.set_title("evaluation_output")
        pred_stat_visualization = self.myscheme.new_node(
            "mlstudiosdk.modules.components.visualization.data_statistics.Statistics")
        pred_stat_visualization.set_title("pred_statistics_visualization")
        pred_stat_writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.JsonWriter")
        pred_stat_writer.set_title("pred_statistics_output")

        self.myscheme.new_link(reader2, "Data", pred_stat_visualization, "Data")
        self.myscheme.new_link(pred_stat_visualization, "Data", pred_stat_writer, "Data")

        self.myscheme.new_link(cnn, "Evaluation Results", eva_visualization, "Result")
        self.myscheme.new_link(cnn, "Metric Score", eva_visualization, "Metric Score")
        # self.myscheme.new_link(sentimentModel, "Metric", eva_visualization, "Metric")
        self.myscheme.new_link(eva_visualization, "Evaluation", evaluation_writer, "Data")

        # self.myscheme.new_link(reader1, "Data", impute, "Train Data")
        # self.myscheme.new_link(reader2, "Data", impute, "Test Data")
        # self.myscheme.new_link(impute, "Train Data", cnn, "Train Data")
        # self.myscheme.new_link(impute, "Test Data", cnn, "Test Data")
        self.myscheme.new_link(reader1, "Data", cnn, "Train Data")
        self.myscheme.new_link(reader2, "Data", cnn, "Test Data")
        self.myscheme.new_link(cnn, "News", outputwriter, "Data")

