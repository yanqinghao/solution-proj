from mlstudiosdk.solution_gallery.solution.SolutionBase import SolutionBase

class AnomalyDetector(SolutionBase):

    def __init__(self):
        super().__init__()
        self.name = "Anomaly Detector"
        self.category_ = 'Automation'
        self.model()

    def model(self):
        reader1 = self.myscheme.new_node("mlstudiosdk.modules.components.io.reader.Reader")
        reader1.set_title("train_input")

        algo = self.myscheme.new_node('model.AnomalyDetector')
        algo.set_title('algorithm')

        TSNE = self.myscheme.new_node('mlstudiosdk.modules.components.visualization.TSNE.TSNE')
        TSNE.set_title("TSNE")

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

        self.myscheme.new_link(reader1, "Data", algo, "Train Data")
        self.myscheme.new_link(algo, "Annotated Data", TSNE, "Annotated Data")

        self.myscheme.new_link(algo, "Annotated Data", outputwriter, "Data")
        self.myscheme.new_link(algo, "Annotated Data", pred_stat_visualization, "Data")
        self.myscheme.new_link(pred_stat_visualization, "Data", pred_stat_writer, "Data")
        self.myscheme.new_link(TSNE, "low_dim Data", evaluation_writer, "Data")
