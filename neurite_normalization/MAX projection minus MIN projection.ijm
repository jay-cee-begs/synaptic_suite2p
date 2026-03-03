//asks user for directory of images
inputDir = getDirectory("Please choose the directory containing your Videos"); 
// Prompt User for z Projection  
//zProj = getString("Enter Projection type:\nMax Intensity, Average Intensity, Min Intensity\nStandard Deviation, Summed Slice, Median", "Average Intensity");
//creates subdirectory for processed images in the same folder
splitDir = inputDir + "/Z-projections/"
File.makeDirectory(splitDir); 
filelist = getFileList(inputDir); 
print("\\Clear"); //clears the log to make progress visible for user in realtime
run("Bio-Formats Macro Extensions");//allows for Ext commands of Bio format to be called from the macro see line 20
setBatchMode(true);

function processImagesInDir(dir) {
	filelist = getFileList(dir);
	// Loop through each file in the directory
    for (i = 0; i < filelist.length; i++) {
        filePath = dir + filelist[i];
        if (File.isDirectory(filePath)) {
            // If it's a directory, process images in the subdirectory
            processImagesInDir(filePath + "/");
        } else if (endsWith(filelist[i], ".nd2")) {
        	processImage(filePath, filelist[i]);

        }
    }
}


function processImage(imagePath, imageName) {
	baseName = replace(imageName, ".nd2", "");
	run("Bio-Formats Importer", "open=[" + imagePath + "] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack");
	imgName = getTitle();
	selectImage(imgName);
	///attempt to check for z-stack
	zStack = nSlices();
	if (zStack > 1){
		
		run("Z Project...", "projection=[Max Intensity]"); 
		 selectWindow(imgName);
		 run("Z Project...", "projection=[Min Intensity]");
		 
         imageCalculator("Subtract create", "MAX_" + imgName, "MIN_" + imgName);
        
         selectWindow("Result of MAX_" + imgName);
         
         saveAs("Tiff",  splitDir + baseName + "_Projection"); 
        
         run("Close All"); 
     		print("Finished processing "+imgName);
     		} else {
    		 write("Task Completed");
    		 waitForUser("Task completed", "The resulting .tiff Images can be found in a new sub folder of your Video folder");
     }
		
	}
	run("Close All"); 


processImagesInDir(inputDir);
setBatchMode(false);
//saveAs("Results", splitDir + "AVG_Skeleton_and_Area_coverage.csv");

//for (i=0; i<filelist.length; i++) { 
//     if (endsWith(filelist[i], ".nd2")){ 
//		
//		Ext.Bio-run("Bio-Formats Importer"(inputDir+filelist[i]));
//		run("Bio-Formats Importer", "open=[inputDir+filelist[i]] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack");
//
//run("Z Project...", "projection=[Average Intensity]");
////selectImage(imgName);
////run("Z Project...", "projection=[Min Intensity);
//
////run("Image Calculator...");
//
////imageCalculator("Subtract create", "MAX_" + imgName","MIN_" + imgName);
//selectImage("AVG_" + imgName);
//         saveAs("Tiff",  splitDir + baseName + "_CP_Projection"); 
