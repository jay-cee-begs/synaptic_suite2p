//asks user for directory of images
inputDir = getDirectory("Please choose the directory containing your Videos"); 
// Prompt User for z Projection  
//zProj = getString("Enter Projection type:\nMax Intensity, Average Intensity, Min Intensity\nStandard Deviation, Summed Slice, Median", "Average Intensity");
//creates subdirectory for processed images in the same folder
splitDir = inputDir + "/Z-projections"
File.makeDirectory(splitDir); 
filelist = getFileList(inputDir); 
print("\\Clear"); //clears the log to make progress visible for user in realtime
run("Bio-Formats Macro Extensions");//allows for Ext commands of Bio format to be called from the macro see line 20
setBatchMode(true);

function processImagesInDir(dir, rootDir) {
	filelist = getFileList(dir);
	// Loop through each file in the directory
    for (i = 0; i < filelist.length; i++) {
        filePath = dir + filelist[i];
        if (File.isDirectory(filePath)) {
            // If it's a directory, process images in the subdirectory
            processImagesInDir(filePath + File.separator, rootDir);
        } else if (endsWith(filelist[i], ".nd2")) {
        	processImage(filePath, filelist[i], rootDir);

        }
    }
}


function processImage(imagePath, imageName, parentDir) {
	baseName = replace(imageName, ".nd2", "");
	if (!endsWith(rootDir, File.separator)) {
        rootDir = rootDir + File.separator;
    }

    // Derive relative path from rootDir
	relPath = substring(imagePath, lengthOf(rootDir));
    relParts = split(relPath, File.separator);

    // Group folder = the first folder after rootDir
    groupName = relParts[0];
    
	run("Bio-Formats Importer", "open=[" + imagePath + "] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack");
	imgName = getTitle();
	selectImage(imgName);
	///attempt to check for z-stack
	zStack = nSlices();
	if (zStack > 1){
		
		run("Z Project...", "projection=[Average Intensity]"); 
        
        selectWindow("AVG_" + imgName);
        print(groupName);
		groupName = replace(groupName, "/", "_");

        print(groupName);
        outName = groupName + baseName + "_Projection.tif";
        print(outName);
        print(groupName);
        savePath = splitDir + File.separator + outName;
        savePath = replace(savePath, "/", File.separator);

        saveAs("Tiff",  savePath); 
         
        print("Finished processing " + baseName + " (Group: " + groupName + ")");

        run("Close All"); 
     
     		} else {
    		 write("Task Completed");
    		 waitForUser("Task completed", "The resulting .tiff Images can be found in a new sub folder of your Video folder");
     }
		
	
	print("Finished processing "+baseName);
	run("Close All"); 
	
}

processImagesInDir(inputDir, inputDir);
setBatchMode(false);
