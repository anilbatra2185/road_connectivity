package test;

import java.io.File;
import java.io.FileOutputStream;

/*
 * Standalone tool to create truth/proposal geojson files from LineStrings.
 * Needed for testing the Python-based scorer.
 */
public class LinestringToGeojson {
	private static String fileName1 = "truth.geojson";
	private static String fileName2 = "solution.geojson";
	private static int id1 = 0;
	private static int id2 = 500;
	
	private static String[] ls1Arr = new String[] {
			"3_edges_connected-move_node_outside_buffer,LINESTRING (500 500, 600 500, 600 700, 550 700)"

	};
	
	private static String[] ls2Arr = new String[] {
			"3_edges_connected-move_node_outside_buffer,LINESTRING (500 500, 600 500, 600 700, 530 700)"
	};
	
	String header = "{\"type\": \"FeatureCollection\",\"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } },\"features\": [\n";
	String footer = "]}";
	String template = "{\"type\": \"Feature\", \"properties\": { \"osm_id\": xxxid, \"type\": \"residential\", \"class\": \"highway\" }, \"geometry\": { \"type\": \"LineString\", \"coordinates\": [ xxxcoord ] } },\n";
	

	private void out(String[] arr, String fileName, int id) throws Exception {
		double scale = 2.7e-6; // This is used in spacenet data. ~ 1 / (111000 / 0.3);
		StringBuilder sb = new StringBuilder();
		sb.append(header);
		for (String line: arr) {
			String lineOut = template;
			lineOut = lineOut.replace("xxxid", "" + id++);
			String coords = "";
			String ls = line.substring(line.indexOf("(") + 1);
			ls = ls.replace(")", "");
			String[] parts = ls.split(",");
			for (String point: parts) {
				point = point.trim();
				String[] xy = point.split(" ");
				double x = Double.parseDouble(xy[0].trim());
				double y = Double.parseDouble(xy[1].trim());
				x = x * scale + 1;
				y = y * scale + 1;
				String coord = "[" + x + ", " + y + ", 0], ";
				coords += coord;
			}
			coords = coords.substring(0, coords.length()-2);
			lineOut = lineOut.replace("xxxcoord", coords);
			sb.append(lineOut);
		}
		sb.append(footer);
		FileOutputStream out = new FileOutputStream(new File(fileName));
		out.write(sb.toString().getBytes());
		out.close();
	}
	
	public static void main(String[] args) throws Exception {
		LinestringToGeojson l2g = new LinestringToGeojson();
		l2g.out(ls1Arr, fileName1, id1);
		l2g.out(ls2Arr, fileName2, id2);
	}
}
