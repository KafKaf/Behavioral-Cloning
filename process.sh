#!/bin/bash

usage() { 
	echo "Usage: $0 [-a <remove_no_angle_records|remove_prefix_from_file>] [-c <main_file,second_file>] [-d <process_directory>] [-r <remove_file>] [-p <remove_prefix>]" 1>&2
       	exit 1
}

remove_prefix_from_file() {
	remove_prefix_escaped=`echo ${remove_prefix} | sed -e 's/\//\\\\\//g'`
	sed -e "s/${remove_prefix_escaped}//g" -i ${remove_file}
}

remove_no_angle() {
	mkdir ${process_directory}/angle_images && mkdir ${process_directory}/no_angle_images
	if [ $? -ne 0 ]; then
		echo "Problem creating new directories"
		exit 1
	fi	

	for line in `cat ${process_directory}/driving_log.csv`
	do 
		steering_angle=`echo $line | awk -F "," '{print $4}'`
		center_image=`echo $line | awk -F "," '{print $1}'`
		left_image=`echo $line | awk -F "," '{print $2}'`
		right_image=`echo $line | awk -F "," '{print $3}'`
		
		if [ "$steering_angle" != "0" ]; then 
			echo $line >> ${process_directory}/new_driving_log.csv
			mv ${process_directory}/${center_image} ${process_directory}/${left_image} ${process_directory}/${right_image} ${process_directory}/angle_images/
		else
			mv ${process_directory}/${center_image} ${process_directory}/${left_image} ${process_directory}/${right_image} ${process_directory}/no_angle_images/
		fi
	done
}

while getopts ":a:d:r:p:" option; do
    case "$option" in
        a)
            action=${OPTARG}
            [[ $action == "remove_no_angle_records" || $action == "remove_prefix_from_file" ]] || usage
            ;;
	d)  process_directory=${OPTARG}
	    if [ ! -d ${process_directory} ]; then
		echo "Directory ${process_directory} doesn't exist, exiting...."
		usage
	    fi	    
	    ;;
	r)
	    remove_file=${OPTARG}
	    if [ ! -f ${remove_file} ]; then
                echo "No files, exiting...."
                usage
            fi
	    ;;
	p)
	   remove_prefix=${OPTARG}
	   ;;
        *)
            usage
            ;;
    esac
done
#shift $((OPTIND-1))

if [ -z $action ]; then
	usage
fi

if [[ $action == "remove_no_angle_records" && -z ${process_directory} ]]; then
	usage
fi

if [[ $action == "remove_prefix_from_file" && -z ${remove_file} && -z ${remove_prefix} ]]; then
	usage
fi

if [ $action == "remove_no_angle_records" ]; then
	remove_no_angle
fi

if [ $action == "remove_prefix_from_file" ]; then
        remove_prefix_from_file
fi
