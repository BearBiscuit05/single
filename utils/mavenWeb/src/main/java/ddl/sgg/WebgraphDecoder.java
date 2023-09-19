package ddl.sgg;

import it.unimi.dsi.fastutil.ints.IntArrayFIFOQueue;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.webgraph.GraphClassParser;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.LazyIntIterator;
import it.unimi.dsi.webgraph.BVGraph;
import java.io.*;
import java.util.*;

public class WebgraphDecoder {
    static private void writeIntLittleEndian(DataOutputStream s, int a) throws Exception{
        s.writeByte(a & 0xFF);
        s.writeByte((a >> 8) & 0xFF);
        s.writeByte((a >> 16) & 0xFF);
        s.writeByte((a >> 24) & 0xFF);
    }
    static public void main(String arg[]) throws Exception {
        ImmutableGraph graph = ImmutableGraph.load(arg[0]);
        // BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(arg[0] + "_coo_edges.bin"));
        BufferedWriter writer = new BufferedWriter(new FileWriter(arg[0] + "_coo_edges.txt"));
        // DataOutputStream dos = new DataOutputStream(bos);
        int num_v = graph.numNodes();
        System.out.printf("Vertices: %d\n", num_v);
        System.out.printf("Edges: %d\n", graph.numArcs());

        long num_e = 0;
        double cur_percent = 0;
        long total_graph_numedge=graph.numArcs();

        int max_node_id = 0;

        for (int v = 0; v < num_v; ++v) {
            LazyIntIterator successors = graph.successors(v);
            for (long i = 0; i < graph.outdegree(v); ++i) {
                int w = successors.nextInt();
                String line = v + "\t" + w; // 将 v 和 w 以制表符分隔的字符串形式连接
                writer.write(line); // 写入到文件
                writer.newLine(); // 写入换行符
                // writeIntLittleEndian(dos, v);
                // writeIntLittleEndian(dos, w);
                if (max_node_id < v) max_node_id = v;
                if (max_node_id < w) max_node_id = w;
                ++num_e;
                if (((double)num_e) * 100 / total_graph_numedge > cur_percent + 5f) {
                    cur_percent = ((double)num_e) * 100 / total_graph_numedge;
                    System.out.printf("%d/%d, %f%% done, cur max node is %d\n", num_e, total_graph_numedge, cur_percent, max_node_id);
                }
            }
        }
        System.out.printf("%d/%d, %f%% done, cur max node is %d\n", num_e, total_graph_numedge, cur_percent, max_node_id);

        // dos.flush();
        // dos.close();
        writer.flush();
        writer.close();
        System.out.printf("Output Edges: %d\n", num_e);
    }
}